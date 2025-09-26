from transformers import AutoModelForCausalLM, AutoTokenizer
from googletrans import Translator

model_name = "facebook/mbart-large-50"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
translator = Translator()

def generate_response(user_input, target_language):
    translated_input = translator.translate(user_input, dest='en').text
    inputs = tokenizer.encode(translated_input, return_tensors="pt")
    outputs = model.generate(inputs, max_length=150, num_beams=5, early_stopping=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    translated_response = translator.translate(response, dest=target_language).text

    return translated_response

def chat():
    print("Welcome to the multi-lingual translator bot! (Type 'quit' to exit)")
    languages = {
    'English': 'en',
    'Spanish': 'es',
    'French': 'fr',
    'German': 'de',
    'Chinese': 'zh-cn',
    'Hindi': 'hi',
    'Arabic': 'ar'
    }
    print('Available languages:')
    for lang, code in languages.items():
        print(f'{code} : {lang}')
    language = input("Select your language: ").lower()

    while True:
        user_input = input("You: ")

        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        bot_response = generate_response(user_input, language)
        print(f"Bot: {bot_response}")

if __name__ == '__main__':
    chat()

