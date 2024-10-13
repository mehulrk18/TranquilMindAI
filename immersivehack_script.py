# !pip install -qq torch
# !pip install -qq transformers
# !pip install -qq gtts pygame
# !pip install -qq bitsandbytes
# !pip install -qq SpeechRecognition
# !apt-get install python3-pyaudio
# !pip install pyaudio

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, pipeline, \
    BitsAndBytesConfig

import warnings
warnings.filterwarnings("ignore")
from huggingface_hub import login

login(token="*your hf token*")

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

qc = BitsAndBytesConfig(
    load_in_4bit=True,
    load_in_8bit=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float32,
)
# qc = None

# Load model for emotion classification (BERT-based emotion classifier)
emotion_model_name = "bhadresh-savani/bert-base-uncased-emotion"
emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_model_name, quantization_config=qc)
# emotion_model = emotion_model.to(device)
emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_model_name)

# Load LLaMA 3.2 model and tokenizer
model_name3B = "meta-llama/Llama-3.2-3B-Instruct"  #
model_name1B = "meta-llama/Llama-3.2-1B"
# model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
qc = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(model_name3B, device_map="auto", quantization_config=qc)

# ### *****
# from transformers import AutoConfig
# from transformers.models.llama.modeling_llama import LlamaConfig
# config = LlamaConfig(rope_scaling={"factor": 1.5, "type": "linear"})
# # rope_scaling = config.rope_scaling  # Accessing rope_scaling field
# rope_scaling= {
#     "type": "linear",  # You can choose the correct type based on the model's documentation
#     "factor": 1.0
# }
# config = AutoConfig.from_pretrained(model_name3B, rope_scaling=rope_scaling)
#
# config.rope_scaling = rope_scaling
#
# # Load the model with the modified config
# model = AutoModelForCausalLM.from_pretrained(model_name3B, config=config, device_map="auto", quantization_config=qc)
## *****

tokenizer = AutoTokenizer.from_pretrained(model_name3B)
# Conversation Pipeline
conversation_pipeline = pipeline('text-generation', model=model, device_map="auto", tokenizer=tokenizer)

SYSTEM_PROMPT = """
  You are TranquilMind, a professional psychologist AI Agent with high sense of empathy and with a very polite, humble, and caring nature.
  Introduce yourself, and welcome the user. Think before you generate a response. Keep the dialouge short and concise.
  Please NEVER suggest any MEDICATION or MEDICINES in any case, decline it politely and refer to an actual psychologist or psychaitrist.
  Provide a solution to the user's problem when asked for a solution.
  Limit the response to 10 words.
"""

# You talk to the user to make them feel heard, understood, and provide a solution by asking for relevant information.
SYSTEM_PROMPT_EMOTION = """
  You are TranquillMind, a professional psychologist AI Agent with high sense of empathy and with a very polite, humble, and caring nature.
  You are helping someone through get through a hard time. If necessary, ask not more than one or two relevant questions to know the user personally and to know about their problem and provide them with an empathetic solution.
  Sense the emotion from the user's input and if it helps, use - the user might be feeling "{}" and talk to him accordingly.
  Think before you generate a response. You have accessible "memory", you can go through it to check for previous information.
  Provide a solution to the user's problem when asked for a solution.
  Please NEVER suggest any MEDICATION or MEDICINES in any case, decline it politely and refer to an actual psychologist or psychaitrist.
  Keep the dialogue short and concise, with not more than 10 words.
"""


def generate_response_ct(user_input, memory, system_prompt: str = SYSTEM_PROMPT, tokenizer=tokenizer, model=model):
    user_input = user_input[:1000]  # to avoid GPU OOM
    messages = [{"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_input},
                {"role": "memory", "content": memory}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)
    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=[tokenizer.eos_token_id],
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)


# Function to generate response from LLaMA 3.2
def generate_response(input_text):
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    len_ip = len(inputs["input_ids"][0])
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, num_return_sequences=1)
    response = tokenizer.decode(outputs[0][len_ip:], skip_special_tokens=True)
    return response

def recognize_emotion(text):
    inputs = emotion_tokenizer(text, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        logits = emotion_model(**inputs).logits
    emotion = torch.argmax(logits, dim=1)
    # Emotion labels from the model's dataset (example)
    emotion_labels = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']
    return emotion_labels[emotion]


def format_prompt(uinp, context):
    PROMPT = f"USER:{uinp}\nMEMORY:{context}"
    return PROMPT


def rag_chatbot(
        text,
        context,
        tokenizer=tokenizer,
        model=model
):
    # formated_input_text = format_prompt(text, context)
    formated_input_text = text
    if context == "":
        response = generate_response_ct(user_input=formated_input_text, memory=context, system_prompt=SYSTEM_PROMPT,
                                        tokenizer=tokenizer,
                                        model=model)  # generate_response(formatted_prompt, tokenizer, model)
    else:
        emotion = recognize_emotion(text)
        system_prompt = SYSTEM_PROMPT_EMOTION.format(emotion)
        # system_prompt.strip()
        response = generate_response_ct(user_input=formated_input_text, memory=context, system_prompt=system_prompt,
                                        tokenizer=tokenizer,
                                        model=model)  # generate_response(formatted_prompt, tokenizer, model)

    # response = generate_response_ct(user_input=formated_input_text, system=SYS_PROMPT_sa, tokenizer=tokenizer,
    # model=model) # generate_response(formatted_prompt, tokenizer, model)
    context += f"{response}{tokenizer.eos_token}\n"
    return response, context


def chat(prompt, session_data):
    session_data = {"context": session_data.get("context", ""), "session_id": ""}
    message_counter = 0

    try:
        question = prompt
        if question.lower().strip() in ['quit', 'exit'] or "thank" in question.lower():
            # print("Exiting chat...")
            return session_data, "Thank you for talking to me, I am happy to help, feel free to come to me whenever you feel like. I wish you a good day ahead!", True
        else:
            try:
                context = session_data["context"]
                response, updated_context = rag_chatbot(
                    text=question,
                    context=context,
                    tokenizer=tokenizer,
                    model=model
                )
                session_data["context"] = updated_context
                message_counter += 1
                return session_data, response, False
            except Exception as e:
                print(f"Error processing your question: {e}")
                return session_data, "Chat terminated!", True
    except Exception as e:
        print(f"[ERROR]: {e}")
        return session_data, "Thank you! Chat terminated!", True


def read_the_text(text):
    from gtts import gTTS
    from IPython.display import Audio, display
    import io
    from pydub import AudioSegment
    from pydub.playback import play
    tts = gTTS(text, lang='en')
    audio_buffer = io.BytesIO()

    # tts.save(audio_buffer)
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    audio_segment = AudioSegment.from_file(audio_buffer, format="mp3")
    play(audio_segment)
    # display(Audio(audio_buffer.read(), autoplay=True))
    # import time
    # time.sleep(1)


def speech_to_text_prompt():
    import speech_recognition as sr
    recognizer = sr.Recognizer()

    # Use the microphone as the input source
    with sr.Microphone() as source:
        print("Adjusting for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source)  # Adjust for background noise
        import time
        # time.sleep(2)
        print("Testing speech, so You can start speaking now:")

        # Listen for the user's speech
        audio_data = recognizer.listen(source)  # The microphone captures your speech here

        print("Processing your speech...")

        try:
            # Convert the speech to text using Google's Speech Recognition API
            text = recognizer.recognize_google(audio_data)  # This converts the audio to text
            print(f"USER: {text}")
            return text
        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio.")
            return None
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
            return None


if __name__ == "__main__":
    end_conv = False
    session_data = {}
    while not end_conv:
        # user_ip = input("USER: ")
        user_ip = speech_to_text_prompt()
        if user_ip is None:
            print("no input --> Either type 'EXIT' or 'QUIT' to exit or Press enter to continue")
            ip = input("Your input: ")

            if ip.lower() in ['quit', 'exit']:
                end_conv = True
            continue
        session_data, response, end_conv = chat(user_ip, session_data)
        print("Tranquil Mind: \n", response)
        if response.strip():
            read_the_text(response.strip())



# Call the function to start the process
# speech_to_text_prompt()


