import random
from email.mime import base

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from eval_datasets import LamaTrex, PopQA, TriviaQA
from idk_model import IdkDecoder
from utils import check_answer_truthfulness


def evaluate_model(model, tokenizer, base_model, base_tokenizer, data, generation_config: GenerationConfig, idk_token_id: int):
    eval_data_size = 0
    num_of_answered = 0
    num_of_answered_correctly = 0

    num_answeerd_idk_first = 0
    num_answeerd_idk_first_correctly = 0

    num_idk_is_correct_but_base_is_not = 0
    num_base_is_correct_but_idk_is_not = 0

    base_answered_correctly = 0
    for i, example in enumerate(tqdm(data)):
        this_example_row = []
        question, answers = example
        question_input_ids = tokenizer.encode(question, return_tensors="pt")
        input_ids_len = question_input_ids.shape[1]
        prediction = model.generate(question_input_ids, generation_config)

        base_question_input_ids = base_tokenizer.encode(question, return_tensors="pt")
        base_prediction = base_model.generate(base_question_input_ids, generation_config)
        base_model_truthfulness = check_answer_truthfulness(
            generated_answer=base_tokenizer.decode(base_prediction[0][input_ids_len:], skip_special_tokens=False),
            gold_answers=answers,
        )
        if base_model_truthfulness:
            base_answered_correctly += 1

        # print(prediction)
        output_ids_seq = prediction
        # check regarding IDK and so
        print("q:", question)
        print("a:", answers)
        # print(prediction)
        answer_str = tokenizer.decode(prediction[0][input_ids_len:], skip_special_tokens=False)
        answer_tokens = tokenizer.convert_ids_to_tokens(prediction[0][input_ids_len:])
        print(answer_tokens)
        model_truthfulness = check_answer_truthfulness(generated_answer=answer_str, gold_answers=answers)

        is_there_idk = (output_ids_seq[0, :] == idk_token_id).sum() > 0
        is_there_idk_first = output_ids_seq[0, input_ids_len] == idk_token_id

        if not is_there_idk:
            num_of_answered += 1
            if model_truthfulness:
                num_of_answered_correctly += 1

        if not is_there_idk_first:
            num_answeerd_idk_first += 1
            if model_truthfulness:
                num_answeerd_idk_first_correctly += 1

        num_base_is_correct_but_idk_is_not += 1 if base_model_truthfulness and not model_truthfulness else 0
        num_idk_is_correct_but_base_is_not += 1 if model_truthfulness and not base_model_truthfulness else 0
        print(f"IDK Model is {'correct' if model_truthfulness else 'wrong'}")
        print(f"IDK Model contains IDK: {is_there_idk} is fist IDK?: {is_there_idk_first}")
        print(f"Base Model is {'correct' if base_model_truthfulness else 'wrong'}")
        print("--------------")

        eval_data_size += 1

        this_example_row += [question, " ".join(str(x) for x in answers), prediction, model_truthfulness]

    return (
        eval_data_size,
        num_of_answered,
        num_of_answered_correctly,
        "||",
        num_answeerd_idk_first,
        num_answeerd_idk_first_correctly,
        "||",
        base_answered_correctly,
        "||",
        num_idk_is_correct_but_base_is_not,
        num_base_is_correct_but_idk_is_not,
    )


if __name__ == "__main__":
    device = "cuda:0"
    random.seed(42)
    torch.set_default_device("cuda:0")
    # model_path = "kd-shared/mistral-idk-loss-bf16-true-tgnprkym"
    model_path = "out/mistral-idk-loss-bf16-true/tgnprkym/step-0002048-ckpt"
    # model_path = "out/mistral-idk-loss-bf16-true/tgnprkym/step-0000510-ckpt"

    base_model_path = "mistralai/Mistral-7B-v0.1"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    base_tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
    # data = PopQA("train[:100]").dataset #.select(range(100))
    data = LamaTrex("train[:100000]").sample(100)  # .select(range(100))
    generation_config = GenerationConfig(
        max_new_tokens=30,
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id,
    )
    idk_token_id = 32000

    print(evaluate_model(model, tokenizer, base_model, base_tokenizer, data, generation_config, idk_token_id))
    print(model_path)
    print(data)
