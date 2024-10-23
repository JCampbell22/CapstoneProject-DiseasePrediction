import tkinter as tk
from tkinter import Text, Entry, Button, END
import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor
import time  
import threading

class Chatbot:
  def __init__(self,root):
    self.root = root
    self.root.title("Chatbot")
    self.text_area = Text(root, bg="white", width=50, height=20)
    self.text_area.pack()
    self.input_field = Entry(root, width=50)
    self.input_field.pack()
    self.send_button = Button(root, text="Send", command= self.threaded_message)
    self.send_button.pack()
    self.model = None
    self.processor = None
    self.initialize_model()

  def initialize_model(self):
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    self.model = MllamaForConditionalGeneration.from_pretrained(
      model_id,
      torch_dtype=torch.float32,
      device_map = "auto"
    )
    self.processor = AutoProcessor.from_pretrained(model_id)

  def llama_response(self, user_input):
    messages = [
      {"role": "user", "content": [
        {"type": "text", "text": user_input},
      ]}
    ]
    text = self.processor.apply_chat_template(messages, add_generation_prompt = True)
    inputs = self.processor(
      text = text,
      return_tensors = "pt"
    ).to(self.model.device)

    start_time = time.time()
    output = self.model.generate(**inputs, max_new_tokens=30)
    response = self.processor.decode(output[0], skip_special_tokens = True)
    end_time = time.time()
    if "assistant" in response:
      response = response.split("assistant")[1].strip()
    duration = end_time - start_time
    return duration, response

  def threaded_message(self):
    user_message = self.input_field.get()
    if user_message:
      self.input_field.delete(0, END)
      self.text_area.configure(state='normal')
      self.text_area.insert(tk.END, "Your: " + user_message + "\n")
      self.text_area.insert(tk.END, "Bot: Thinking...\n")
      self.text_area.configure(state="disabled")
    threading.Thread(target=self.generate_response, args=(user_message,)).start()

  def generate_response(self,user_message):
    try:
      duration, response = self.llama_response(user_message)
    except Exception as e:
      duration, response = f"Error: {str(e)}", 0.0

    self.update_message(f"\nBot: {response} (Response time: {duration:.2f} seconds.)\n")

  def update_message(self, message):
    self.text_area.configure(state='normal')
    self.text_area.delete('end-2l', tk.END)
    self.text_area.insert(tk.END, message)
    self.text_area.configure(state='disabled')
    self.text_area.yview(tk.END)

if __name__ == "__main__":
  root = tk.Tk()
  app = Chatbot(root)
  root.mainloop()
