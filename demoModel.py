import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_questions(context, answer,clue,style):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    input_text = f"context: {context} answer: {answer} clue: {clue} question: {style}"
    input_ids = tokenizer.encode(input_text, add_special_tokens=True, return_tensors="pt")

    # Generar preguntas
    question = model.generate(input_ids=input_ids, max_length=200, num_return_sequences=1, early_stopping=True)

    # Decodificar la pregunta generada
    question = tokenizer.decode(question[0], skip_special_tokens=True)
    question = question.split("question:")[1]
    question = question.split("\n")[0]
    return f"question: {question}"


