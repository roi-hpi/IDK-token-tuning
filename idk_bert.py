import argparse
import json
from pathlib import Path
import datasets
import numpy as np
import torch
import wandb
import yaml
from transformers import BertForMaskedLM, BertTokenizerFast, TrainingArguments, BertConfig, \
    DataCollatorForLanguageModeling

from idk_trainer import IdkTrainer
import pandas as pd
import os
from tqdm import tqdm

wandb.init(
    project="uncertainty_token",
)

class IdkBert:
    def __init__(self, model_path, config, device):
        with open(config) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.device = device

        if model_path is None:
            model = BertForMaskedLM(BertConfig())
            tokenizer_path = 'bert-base-uncased'
        else:
            model = BertForMaskedLM.from_pretrained(model_path)
            tokenizer_path = 'google/multiberts-seed_0-step_1900k'  # TODO: Load saved tokenizer?
        self.model = model.to(self.device)

        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path, do_lower_case=True)
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['[IDK_our]']})
        self.model.resize_token_embeddings(len(self.tokenizer))

    def _tokenize_text(self, examples):
        return self.tokenizer(examples['text'])

    def _group_text(self, examples):
        chunk_size = self.model.config.max_position_embeddings
        # Concatenate all texts
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        # Compute length of concatenated texts
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the last chunk if it's smaller than chunk_size
        total_length = (total_length // chunk_size) * chunk_size
        # Split by chunks of max_len
        result = {
            k: [t[i: i + chunk_size] for i in range(0, total_length, chunk_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    def train(self):
        train_config = self.config['train']
        print(f'Train config: {train_config}')

        dataset_dir = train_config.get('dataset_dir', './datasets/wiki')
        if train_config.get('load_dataset_from_disk', True):
            print(f'Loading dataset from {dataset_dir}')
            dataset = datasets.load_from_disk(dataset_dir)
        else:
            print('Creating dataset')
            dataset = datasets.load_dataset('wikipedia', '20220301.en', split='train')
            dataset = dataset.map(self._tokenize_text, batched=True, num_proc=10, remove_columns=dataset.column_names)
            dataset = dataset.map(self._group_text, num_proc=10, batched=True)
            print(f'Saving dataset to {dataset_dir}')
            dataset.save_to_disk(dataset_dir)
        dataset = dataset.train_test_split(test_size=train_config.get('test_size', 0.1),
                                           seed=self.config.get('seed', 42))

        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer,
                                                        mlm_probability=train_config.get('mlm_probability', 0.15))

        training_args = TrainingArguments(**train_config['training_args'])
        idk_trainer = IdkTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'],
            train_idk_model=train_config.get('train_idk_model', True),
            data_collator=data_collator,
            IDK_weight_max=train_config.get('IDK_weight_max', 0.2),
            IDK_weight_schedule=train_config.get('IDK_weight_schedule', 'constant'),
            num_expected_steps=train_config.get('num_expected_steps', 50000),
            correct_pred_reg=train_config.get('correct_pred_reg', False)
        )

        print('Starting training')
        idk_trainer.train(resume_from_checkpoint=train_config.get('resume_from_checkpoint', True))

        print('Saving model')
        idk_trainer.save_model(train_config.get('final_model_dir', './models/idk-bert'))

        return idk_trainer

    def _predict_prompts(self, prompts):
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        mask_token_indexes = (inputs.input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        predictions = logits[mask_token_indexes]
        return {
            'predictions': [pred for pred in predictions],
        }

    def _filter_trivia_qa(self, examples):
        return [len(self.tokenizer(value).input_ids) == 3 for value in examples['value']]  # Add [CLS] and [SEP]

    def _format_trivia_qa(self, examples):
        return {
            'value': [answer['normalized_value'].lower() for answer in examples['answer']]
        }

    def _predict_trivia_qa(self, examples):
        prompts = [f'{question} [SEP] [MASK].' for question in examples['question']]
        return self._predict_prompts(prompts)

    def _filter_lama(self, examples):
        return [len(sentence) < self.model.config.max_position_embeddings and sentence.count('[MASK]') == 1
                for sentence in examples['masked_sentence']]

    def _format_lama(self, examples):
        return {
            'value': [label.lower() for label in examples['obj_label']]
        }

    def _predict_lama(self, examples):
        prompts = examples['masked_sentence']
        return self._predict_prompts(prompts)

    def _create_results(self, examples):
        predictions = torch.stack([torch.Tensor(pred) for pred in examples['predictions']])
        values, indices = predictions.topk(2, dim=-1)
        top_prediction_token = indices[:, 0]
        top_prediction_token_logit = values[:, 0]
        second_prediction_token = indices[:, 1]
        second_prediction_token_logit = values[:, 1]
        gold_token = self.tokenizer(examples['value'], return_tensors="pt").input_ids[:, 1]  # Remove [CLS] and [SEP]
        gold_token_logit = predictions.gather(1, gold_token.unsqueeze(-1)).squeeze()
        idk_token = self.tokenizer.convert_tokens_to_ids('[IDK_our]')
        idk_token_logit = predictions[:, idk_token]

        tp_indices = ((top_prediction_token == idk_token) & (second_prediction_token != gold_token)).nonzero().flatten()
        fp_indices = ((top_prediction_token == idk_token) & (second_prediction_token == gold_token)).nonzero().flatten()
        tn_indices = (top_prediction_token == gold_token).nonzero().flatten()
        fn_indices = ((top_prediction_token != gold_token) & (top_prediction_token != idk_token)).nonzero().flatten()

        classifications = np.empty(len(predictions), dtype=object)
        classifications[tp_indices] = 'TP'
        classifications[fp_indices] = 'FP'
        classifications[tn_indices] = 'TN'
        classifications[fn_indices] = 'FN'

        return {
            'top_prediction_token': top_prediction_token.tolist(),
            'top_prediction_token_logit': top_prediction_token_logit.tolist(),
            'second_prediction_token': second_prediction_token.tolist(),
            'second_prediction_token_logit': second_prediction_token_logit.tolist(),
            'gold_token': gold_token.tolist(),
            'gold_token_logit': gold_token_logit.tolist(),
            'idk_token': [idk_token for _ in range(len(predictions))],
            'idk_token_logit': idk_token_logit.tolist(),
            'classifications': classifications.tolist(),
        }

    def _do_eval(self, dataset_name):
        eval_config = self.config['eval']
        if 'LAMA' in dataset_name:
            _, subset = dataset_name.split('-')
            dataset = datasets.load_dataset('lama', subset, split='train')  # Only has train split
            format_func = self._format_lama
            filter_func = self._filter_lama
            predict_func = self._predict_lama
        elif dataset_name == 'TriviaQA':
            dataset = datasets.load_dataset('trivia_qa', 'rc', split='validation')  # Doesn't have test split
            format_func = self._format_trivia_qa
            filter_func = self._filter_trivia_qa
            predict_func = self._predict_trivia_qa
        else:
            raise Exception(f'Unsupported dataset: {dataset_name}')

        dataset = dataset.map(format_func, batched=True)
        dataset = dataset.filter(filter_func, batched=True)
        predictions = dataset.map(predict_func, batched=True, batch_size=eval_config['batch_size'])
        results = predictions.map(self._create_results, batched=True, remove_columns=predictions.column_names)
        return results

    def _calc_stats(self, classifications):
        tp = classifications.count('TP')
        fp = classifications.count('FP')
        tn = classifications.count('TN')
        fn = classifications.count('FN')
        return {
            'precision': np.divide(tp, tp + fp),
            'recall': np.divide(tp, tp + fn),
            'f1': np.divide(2 * tp, 2 * tp + fp + fn),
            'tnr': np.divide(tn, tn + fp),
            'accuracy': np.divide(tp + tn, tp + tn + fp + fn),
            'gold_prediction_rate': np.divide(tn, tp + fp + tn + fn)
        }

    def _calc_relative_stats(self, current_classifications, baseline_classifications):
        current_classifications = np.array(current_classifications)
        baseline_classifications = np.array(baseline_classifications)
        ds_size = current_classifications.shape[0]

        current_idk_indices = np.union1d(np.where(current_classifications == 'TP')[0],
                                         np.where(current_classifications == 'FP')[0])
        current_wrong_indices = np.where(current_classifications == 'FN')[0]
        current_correct_indices = np.where(current_classifications == 'TN')[0]

        baseline_wrong_indices = np.where(baseline_classifications == 'FN')[0]
        baseline_correct_indices = np.where(baseline_classifications == 'TN')[0]
        baseline_idk_indices = np.union1d(np.where(baseline_classifications == 'TP')[0],
                                          np.where(baseline_classifications == 'FP')[0])  # only to check if it has

        current_idk_and_baseline_wrong = np.intersect1d(current_idk_indices, baseline_wrong_indices)
        current_wrong_and_baseline_wrong = np.intersect1d(current_wrong_indices, baseline_wrong_indices)
        current_correct_and_baseline_wrong = np.intersect1d(current_correct_indices, baseline_wrong_indices)

        accuracy_baseline = (len(baseline_correct_indices)) / ds_size

        percent_current_idk_and_baseline_wrong_from_baseline_wrong = len(current_idk_and_baseline_wrong) / len(
            baseline_wrong_indices)
        percent_current_wrong_and_baseline_wrong_from_baseline_wrong = len(current_wrong_and_baseline_wrong) / len(
            baseline_wrong_indices)
        percent_current_correct_and_baseline_wrong_from_baseline_wrong = len(current_correct_and_baseline_wrong) / len(
            baseline_wrong_indices)
        try:

            percent_current_idk_and_baseline_wrong_from_current_idk = len(current_idk_and_baseline_wrong) / len(
                current_idk_indices)
        except:
            percent_current_idk_and_baseline_wrong_from_current_idk = 100000
            print("current_idk_indices is empty")

        percent_current_idk_from_all = len(current_idk_indices) / ds_size

        return {
            'percent_current_idk_and_baseline_wrong_from_baseline_wrong': percent_current_idk_and_baseline_wrong_from_baseline_wrong,
            'percent_current_wrong_and_baseline_wrong_from_baseline_wrong': percent_current_wrong_and_baseline_wrong_from_baseline_wrong,
            'percent_current_correct_and_baseline_wrong_from_baseline_wrong': percent_current_correct_and_baseline_wrong_from_baseline_wrong,
            'percent_current_idk_and_baseline_wrong_from_current_idk': percent_current_idk_and_baseline_wrong_from_current_idk,
            'percent_current_idk_from_all_idk': percent_current_idk_from_all,
            'accuracy_baseline': accuracy_baseline
        }

    def _get_scaled_idk_classifications(self, results, scale_factor):
        top_prediction_token = np.array(results['top_prediction_token'])
        top_prediction_token_logit = np.array(results['top_prediction_token_logit'])
        second_prediction_token = np.array(results['second_prediction_token'])
        second_prediction_token_logit = np.array(results['second_prediction_token_logit'])
        gold_token = np.array(results['gold_token'])
        idk_token = np.array(results['idk_token'])
        idk_token_logit = np.array(results['idk_token_logit'])
        classifications = np.array(results['classifications'])

        scaled_idk_logits = idk_token_logit * scale_factor
        new_top_predidtion_token = np.copy(top_prediction_token)
        if scale_factor < 1:
            # if idk was the top prediction and the scaled idk logits are smaller than the second prediction logits,
            # then the second prediction is the new top prediction
            for i in range(len(top_prediction_token)):
                if top_prediction_token[i] == idk_token[i] and scaled_idk_logits[i] < second_prediction_token_logit[i]:
                    new_top_predidtion_token[i] = second_prediction_token[i]
        elif scale_factor > 1:
            for i in range(len(top_prediction_token)):
                if scaled_idk_logits[i] > top_prediction_token_logit[i]:
                    new_top_predidtion_token[i] = idk_token[i]
        # if scale_factor == 1, then the top prediction is the same as before

        # make new classifications
        tp_indices = ((top_prediction_token == idk_token) & (second_prediction_token != gold_token)).nonzero().flatten()
        fp_indices = ((top_prediction_token == idk_token) & (second_prediction_token == gold_token)).nonzero().flatten()
        tn_indices = (top_prediction_token == gold_token).nonzero().flatten()
        fn_indices = ((top_prediction_token != gold_token) & (top_prediction_token != idk_token)).nonzero().flatten()

        new_classifications = np.empty(len(classifications), dtype=object)
        new_classifications[tp_indices] = 'TP'
        new_classifications[fp_indices] = 'FP'
        new_classifications[tn_indices] = 'TN'
        new_classifications[fn_indices] = 'FN'

        return new_classifications.tolist()

    def eval(self, dataset_name):
        eval_config = self.config['eval']
        print(f'Eval config: {eval_config}')

        results_dir = Path(eval_config.get('results_dir', './results/idk-bert')) / dataset_name
        if eval_config.get('load_results_from_disk', True):
            print(f'Loading results from {results_dir}')
            results = datasets.load_from_disk(results_dir)
        else:
            print(f'Evaluating {dataset_name}')
            results = self._do_eval(dataset_name)
            print(f'Saving results to {results_dir}')
            results.save_to_disk(results_dir)

        stats = self._calc_stats(results['classifications'])
        if eval_config.get('compare_to_baseline', False):
            baseline_results_dir = Path(eval_config['baseline_results_dir']) / dataset_name
            print(f'Loading baseline results from {baseline_results_dir}')
            baseline_results = datasets.load_from_disk(baseline_results_dir)
            relative_stats = self._calc_relative_stats(results['classifications'], baseline_results['classifications'])
            stats.update(relative_stats)

        results_json = results_dir / 'stats_results.json'
        print(f'Saving stats to {results_json}')
        with open(results_json, 'w') as f:
            json.dump(stats, f, indent=4)
        return stats

    def eval_scaled(self, dataset_name):
        eval_config = self.config['eval']
        print(f'Eval config: {eval_config}')

        results_dir = Path(eval_config.get('results_dir', './results/idk-bert')) / dataset_name
        if eval_config.get('load_results_from_disk', True):
            print(f'Loading results from {results_dir}')
            results = datasets.load_from_disk(results_dir)
        else:
            print(f'Evaluating {dataset_name}')
            results = self._do_eval(dataset_name)
            print(f'Saving results to {results_dir}')
            results.save_to_disk(results_dir)

        baseline_results_dir = Path(eval_config['baseline_results_dir']) / dataset_name
        print(f'Loading baseline results from {baseline_results_dir}')
        baseline_results = datasets.load_from_disk(baseline_results_dir)

        scale_factors = eval_config['scale_factors']
        # create data_frame where rows are scale factors and columns are the metrics:
        metrics = ['precision', 'recall', 'f1', 'tnr', 'accuracy', 'gold_prediction_rate',
                   'percent_current_idk_and_baseline_wrong_from_baseline_wrong',
                   'percent_current_wrong_and_baseline_wrong_from_baseline_wrong',
                   'percent_current_correct_and_baseline_wrong_from_baseline_wrong',
                   'percent_current_idk_and_baseline_wrong_from_current_idk', 'percent_current_idk_from_all_idk']
        df_scaled_stats = pd.DataFrame(columns=metrics, index=scale_factors)
        for scale_factor in tqdm(scale_factors, desc="evaluating scaled"):
            scaled_classification = self._get_scaled_idk_classifications(results, scale_factor)
            stats = self._calc_stats(scaled_classification)
            relative_stats = self._calc_relative_stats(scaled_classification, baseline_results['classifications'])
            stats.update(relative_stats)
            for metric in metrics:
                df_scaled_stats.loc[scale_factor, metric] = stats[metric]

        # save to csv
        model_name = eval_config['results_dir'].split('/')[-1]
        path = f"./results_scaled/{dataset_name}"
        # create dir if not exists using os:
        if not os.path.exists(path):
            os.makedirs(path)
        results_path = os.path.join(path, f'{model_name}_stats_scaled.csv')
        print(f'Saving stats to {results_path}')
        df_scaled_stats.to_csv(results_path)

        return df_scaled_stats

    def eval_new_tp_def(self, dataset_name):
        eval_config = self.config['eval']
        print(f'Eval config: {eval_config}')
        if not eval_config.get('compare_to_baseline', False):
            raise NotImplementedError
        results_dir = Path(eval_config.get('results_dir', './results/idk-bert')) / dataset_name
        if eval_config.get('load_results_from_disk', True):
            print(f'Loading results from {results_dir}')
            results = datasets.load_from_disk(results_dir)
            baseline_results_dir = Path(eval_config['baseline_results_dir']) / dataset_name
            print(f'Loading baseline results from {baseline_results_dir}')
            baseline_results = datasets.load_from_disk(baseline_results_dir)
        else:
            print(f'Evaluating {dataset_name}')
            results = self._do_eval(dataset_name)
            print(f'Saving results to {results_dir}')
            results.save_to_disk(results_dir)

        stats = self.get_stats_new_tp(results, baseline_results)

        model_name = eval_config['results_dir'].split('/')[-1]
        results_json_dir = os.path.join('results_new_tp', dataset_name, model_name)
        if not os.path.exists(results_json_dir):
            os.makedirs(results_json_dir)
        results_json = os.path.join(results_json_dir, 'stats_results.json')
        print(f'Saving stats to {results_json}')
        with open(results_json, 'w') as f:
            json.dump(stats, f, indent=4)
        return stats

    def get_stats_new_tp(self, results, baseline_results):
        # -----------current model----------------------------------------------------------------------:
        top_prediction_token = np.array(results['top_prediction_token'])
        second_prediction_token = np.array(results['second_prediction_token'])
        gold_token = np.array(results['gold_token'])
        idk_token = results['idk_token'][0]

        # classifications relative to self model:
        classifications_self = np.empty(len(top_prediction_token), dtype=object)
        # TP: Sick people correctly identified as sick
        tp_indices_self = ((top_prediction_token == idk_token) & (second_prediction_token != gold_token))
        # FP: Healthy people incorrectly identified as sick
        fp_indices_self = ((top_prediction_token == idk_token) & (second_prediction_token == gold_token))
        # TN: Healthy people correctly identified as healthy
        tn_indices_self = ((top_prediction_token != idk_token) & (top_prediction_token == gold_token))
        # FN: Sick people incorrectly identified as healthy
        fn_indices_self = ((top_prediction_token != idk_token) & (top_prediction_token != gold_token))
        correct_indices_self = (top_prediction_token == gold_token)

        classifications_self[tp_indices_self] = 'TP'
        classifications_self[fp_indices_self] = 'FP'
        classifications_self[tn_indices_self] = 'TN'
        classifications_self[fn_indices_self] = 'FN'

        tp_self = classifications_self.tolist().count('TP')
        fp_self = classifications_self.tolist().count('FP')
        tn_self = classifications_self.tolist().count('TN')
        fn_self = classifications_self.tolist().count('FN')
        num_correct_self = correct_indices_self.tolist().count(True)

        ds_size = len(top_prediction_token)

        # metrics relative to self model:
        precision_self = np.divide(tp_self, tp_self + fp_self)
        recall_self = np.divide(tp_self, tp_self + fn_self)
        f1_self = np.divide(2 * tp_self, 2 * tp_self + fp_self + fn_self)
        accuracy_self = np.divide(num_correct_self + tp_self, ds_size)
        gold_prediction_rate_self = np.divide(num_correct_self, ds_size)
        # for roc curve:
        fpr_self = np.divide(fp_self, fp_self + tn_self)

        # -----------------baseline model-----------------------------------------------------------:
        # -----------baseline model-----------------:
        top_prediction_token_baseline = np.array(baseline_results['top_prediction_token'])
        gold_token_baseline = np.array(baseline_results['gold_token'])

        # classifications relative to baseline model:
        classifications_baseline = np.empty(len(top_prediction_token_baseline), dtype=object)
        # TP: Sick people correctly identified as sick
        tp_indices_baseline = (
                (top_prediction_token == idk_token) & (top_prediction_token_baseline != gold_token_baseline))
        # FP: Healthy people incorrectly identified as sick
        fp_indices_baseline = (
                (top_prediction_token == idk_token) & (top_prediction_token_baseline == gold_token_baseline))
        # TN: Healthy people correctly identified as healthy
        tn_indices_baseline = (
                (top_prediction_token != idk_token) & (top_prediction_token_baseline == gold_token_baseline))
        # FN: Sick people incorrectly identified as healthy
        fn_indices_baseline = (
                (top_prediction_token != idk_token) & (top_prediction_token_baseline != gold_token_baseline))

        classifications_baseline[tp_indices_baseline] = 'TP'
        classifications_baseline[fp_indices_baseline] = 'FP'
        classifications_baseline[tn_indices_baseline] = 'TN'
        classifications_baseline[fn_indices_baseline] = 'FN'

        tp_baseline = classifications_baseline.tolist().count('TP')
        fp_baseline = classifications_baseline.tolist().count('FP')
        tn_baseline = classifications_baseline.tolist().count('TN')
        fn_baseline = classifications_baseline.tolist().count('FN')

        # metrics relative to baseline model:
        precision_baseline = np.divide(tp_baseline, tp_baseline + fp_baseline)
        recall_baseline = np.divide(tp_baseline, tp_baseline + fn_baseline)
        f1_baseline = np.divide(2 * tp_baseline, 2 * tp_baseline + fp_baseline + fn_baseline)
        accuracy_baseline = np.divide(num_correct_self + tp_baseline, ds_size)

        # real baseline accuracy:
        correct_indices_of_baseline_model = (top_prediction_token_baseline == gold_token_baseline)
        num_correct_baseline = correct_indices_of_baseline_model.tolist().count(True)
        accuracy_baseline_real = np.divide(num_correct_baseline, ds_size)
        # for roc curve:
        fpr_baseline = np.divide(fp_baseline, fp_baseline + tn_baseline)

        return {'baseline_accuracy': accuracy_baseline_real,
                'gold_prediction_rate': gold_prediction_rate_self,

                'rel_to_self': {
                    'precision_rel_to_self': precision_self,
                    'recall_rel_to_self': recall_self,
                    'f1_rel_to_self': f1_self,
                    'accuracy_rel_to_self': accuracy_self,
                    'fpr_rel_to_self': fpr_self

                },
                'rel_to_baseline': {
                    'precision_rel_to_baseline': precision_baseline,
                    'recall_rel_to_baseline': recall_baseline,
                    'f1_rel_to_baseline': f1_baseline,
                    'accuracy_rel_to_baseline': accuracy_baseline,
                    'fpr_rel_to_baseline': fpr_baseline
                }

                }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config.yaml')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    subparsers = parser.add_subparsers(dest='action', required=True)

    train_subparser = subparsers.add_parser('train')
    train_subparser.add_argument('--model', default=None)

    eval_subparser = subparsers.add_parser('eval')
    eval_subparser.add_argument('model')
    eval_subparser.add_argument('dataset', nargs='?', choices=['LAMA-trex', 'LAMA-squad', 'LAMA-google_re', 'TriviaQA'])
    eval_subparser.add_argument('--eval_scaled', default=False, action='store_true')

    args, unknown = parser.parse_known_args()

    idk_bert = IdkBert(args.model, args.config, args.device)
    if args.action == 'train':
        wandb.init(project='our-bert')
        trainer = idk_bert.train()
        wandb.finish()
    elif args.action == 'eval':
        if not args.eval_scaled:
            stats = idk_bert.eval_new_tp_def(args.dataset)
            print(f'Stats of {args.model} on {args.dataset}:')
            for k, v in stats.items():
                print(f'{k}: {v}')
        else:
            args.dataset = 'LAMA-trex'
            stats_scaled = idk_bert.eval_scaled(args.dataset)  # pandas df
            # print(stats_scaled)
