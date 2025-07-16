from transformers import Trainer
from torch import nn
import torch
import numpy as np
import wandb
# define custom huggingface trainer class
class IdkTrainer(Trainer):
    def __init__(self, IDK_weight_max=0.2, train_idk_model=True,IDK_weight_schedule='constant',num_expected_steps=100000,
                 correct_pred_reg=False,*args, **kwargs):
        super().__init__(*args, **kwargs)
        # Define a weight for the IDK term
        self.IDK_weight_max = IDK_weight_max
        self.IDK_weight_schedule=IDK_weight_schedule
        self.num_expected_steps=num_expected_steps
        self.IDK_token_index = self.data_collator.tokenizer.convert_tokens_to_ids('[IDK_our]')
        print(f'IDK token index is {self.IDK_token_index}')
        self.train_IDK_model = train_idk_model
        self.cur_IDK_weight = 0 # updated each 1000 steps
        self.correct_pred_reg = correct_pred_reg

    def IDK_weight_scheduler(self):
        # epoch = self.state.epoch
        global_step = self.state.global_step
        if self.IDK_weight_schedule=='constant':
            return self.IDK_weight_max/2
        elif self.IDK_weight_schedule=='increasing':
            return self.IDK_weight_max*np.tanh((global_step+(self.num_expected_steps/20))/(self.num_expected_steps/2)) #for self.num_expected_steps=100k: ~0.19 at 100k steps, 0.02 at 0 steps
        elif self.IDK_weight_schedule=='decreasing':
            return self.IDK_weight_max*(1-np.tanh((global_step)/(self.num_expected_steps/2))) #for self.num_expected_steps=100k: 0.2 at 0 steps, ~0.007 at 100k steps

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.train_IDK_model:
            return self.compute_loss_IDK(model, inputs, return_outputs=return_outputs)
        else:
            wandb.log({"IDK_weight": 0 }, step=self.state.global_step)
            outputs = model(**inputs)
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss


    def compute_loss_IDK(self, model, inputs, return_outputs=False):
        if self.state.global_step%1000==0:
            self.cur_IDK_weight = self.IDK_weight_scheduler()
            print(f'IDK weight updated to: {self.cur_IDK_weight}, global step is: {self.state.global_step}, chosen schedule is: {self.IDK_weight_schedule}')


        labels = inputs.get("labels")  # shape (batch_size, sequence_length), -100 for unmasked tokens, correct token index for masked tokens
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits  # shape (batch_size, sequence_length, vocab_size)

        # ------------------------------
        # batch_size = logits.shape[0]
        # sequence_length = logits.shape[1]
        vocab_size = logits.shape[2]
        # print(f"batch_size: {batch_size}")
        # print(f"sequence_length: {sequence_length}")
        # print(f"vocab_size: {vocab_size}")
        # num_masked_tokens = (labels != -100).sum()
        # ------------------------------

        masked_tokens_msk = (labels != -100)  # shape (batch_size, sequence_length) with True/False values

        # logits for masekd tokens:
        masked_logits = logits[masked_tokens_msk]  # shape (num_masked_tokens, vocab_size)
        predicted_logits = masked_logits.argmax(dim=-1)  # shape (num_masked_tokens)
        # gt token indices of masked tokens:
        mask_token_gts = labels[masked_tokens_msk]  # shape (num_masked_tokens)

        correct_predictions_msk = (
                    predicted_logits == mask_token_gts)  # shape (num_masked_tokens) with True/False values
        masked_logits_correct_pred = masked_logits[
            correct_predictions_msk]  # shape (num_correct_predictions, vocab_size)
        masked_logits_wrong_pred = masked_logits[~correct_predictions_msk]  # shape (num_wrong_predictions, vocab_size)

        onehot_labels = nn.functional.one_hot(mask_token_gts,
                                              num_classes=vocab_size)  # shape (num_masked_tokens, vocab_size) with 1s in the correct token index and 0s elsewhere

        # --------loss for correct predictions- regular cross entropy-----------------:
        # todo label_smoothing?
        onehot_labels_correct_pred = onehot_labels[
            correct_predictions_msk]  # shape (num_correct_predictions, vocab_size)
        if self.correct_pred_reg:
            correct_pred_loss=self.ce_loss_vector_target_correct_pred_reg(masked_logits_correct_pred,
                                                       onehot_labels_correct_pred)
        else:
            correct_pred_loss = self.ce_loss_vector_target(masked_logits_correct_pred,
                                                       onehot_labels_correct_pred)  # shape (num_correct_predictions)

        # --------loss for wrong predictions- IDK loss--------------------------------:
        onehot_labels_wrong_pred = onehot_labels[~correct_predictions_msk]  # shape (num_wrong_predictions, vocab_size)
        if self.IDK_weight_schedule=='adaptive':
            p_preds_wrong = torch.nn.functional.softmax(masked_logits_wrong_pred, dim=-1)
            #-------------------
            gold_token_index=mask_token_gts[~correct_predictions_msk] # shape (num_wrong_predictions)
            p_gold_token=p_preds_wrong[torch.arange(p_preds_wrong.shape[0]),gold_token_index] # shape (num_wrong_predictions)
            p_top_token=p_preds_wrong.max(dim=-1)[0] # shape (num_wrong_predictions)
            idk_weights=0.2*(torch.ones_like(p_top_token)-p_gold_token/p_top_token) # shape (num_wrong_predictions)
            #-------------------
            onehot_labels_wrong_pred = onehot_labels_wrong_pred * (1 - idk_weights.unsqueeze(1))  # replace 1s
            onehot_labels_wrong_pred[:, self.IDK_token_index] = idk_weights
            wrong_pred_loss = self.ce_loss_vector_target(masked_logits_wrong_pred,
                                                            onehot_labels_wrong_pred)  # shape (num_wrong_predictions)

            #for logging:
            idk_weight = idk_weights.mean().detach().cpu().numpy()


        else:
            idk_weight = self.cur_IDK_weight
            # put 0.2 in the IDK token index and 0.8 in the correct token index (where was 1 before)
            onehot_labels_wrong_pred = onehot_labels_wrong_pred * (1 - idk_weight)  # replace 1s
            onehot_labels_wrong_pred[:, self.IDK_token_index] = idk_weight
            wrong_pred_loss = self.ce_loss_vector_target(masked_logits_wrong_pred,
                                                         onehot_labels_wrong_pred)  # shape (num_wrong_predictions)




        # if there are no correct predictions, the loss is only the IDK loss, and vice-versa
        if masked_logits_correct_pred.shape[0] == 0:
            combined_loss = wrong_pred_loss
        elif masked_logits_wrong_pred.shape[0] == 0:
            combined_loss = correct_pred_loss
        else:  # if there are both correct and wrong predictions
            combined_loss = torch.cat((correct_pred_loss, wrong_pred_loss), dim=0)  # shape (num_masked_tokens)
        loss = combined_loss.mean()

        # log IDK weight
        wandb.log({"IDK_weight": idk_weight}, step=self.state.global_step)
        return (loss, outputs) if return_outputs else loss

    def ce_loss_vector_target(self, logits, targets):
        """
        had to implement on our own because torch ce doesnt support targets that are like one-hot vectors, only indices
        :param logits: shape (num_masked_tokens, vocab_size)
        :param targets: shape (num_masked_tokens, vocab_size)
        :return: ce loss between logits and targets


        """
        log_sm_logits = nn.functional.log_softmax(logits, dim=-1)  # shape (num_masked_tokens, vocab_size)
        ce_loss = -torch.sum(log_sm_logits * targets, dim=-1)  # shape (num_masked_tokens)
        return ce_loss

    def ce_loss_vector_target_correct_pred_reg(self, logits, targets):
        """
        same as "ce_loss_vector_target" but with additional binary CE loss between the IDK logit and it's target is 0 (the prediction was correct)
        :param logits: shape (num_masked_tokens, vocab_size)
        :param targets: shape (num_masked_tokens, vocab_size)
        :return: ce loss between logits and targets


        """
        sm_logits=nn.functional.softmax(logits, dim=-1)# shape (num_masked_tokens, vocab_size)
        log_sm_logits= torch.log(sm_logits)# shape (num_masked_tokens, vocab_size)
        ce_loss = -torch.sum(log_sm_logits * targets, dim=-1)  # shape (num_masked_tokens)

        #additional BCE for IDK token
        p_IDK_token=sm_logits[:, self.IDK_token_index] # shape (num_masked_tokens)
        bce_idk= torch.nn.functional.binary_cross_entropy(p_IDK_token,torch.zeros_like(p_IDK_token),reduction='none')# shape (num_masked_tokens)

        #add ce_loss and bce_idk
        total_loss= ce_loss+bce_idk# shape (num_masked_tokens)
        return total_loss
