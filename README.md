# TranquilMindAI
Project Tranquil Mind - for Immersive Reality Hackathon
___
## Description
Imagine your happy place. Your safe space. Got it?
Now imagine you can be transported there with a click of a button and experience the same serenity as you did the last time you were there. Feeling safe yet? Wait, it gets better. Your digital avatar is here! An AI-powered agent who is empowered and equipped to talk to you in real time, react and learn from your interactions, and with increased use, gradually come to understand you on a much deeper level. 
No, this is not The Matrix. This hyper-focussed conversational system is imbibed onto a digital avatar into the environment that the system perceives to be your ‘happy place’. The idea is to leverage XR technology to create a space where users can explore their feelings, resolve inner conflicts, or simply take a moment to breathe. The personalized experience (initially choosing the happy place scene during setup, according to what calms you down; then the adaptive imitation learning performed by the AI agent) respects user privacy, ensuring all interactions remain secure and data is responsibly managed. With a focus on inclusivity, users can customize their avatars to represent themselves, creating a safe, relatable virtual environment. We envision a digital community where technology helps users find connection, understanding, and balance in their daily lives. We imagine this to be a handy tool for people to balance themselves in the middle of a rough day or when things seem especially down with a quick talk with themselves. 
Feeling conflicted about that recent argument with your loved one? Overwhelmed with work and unsure of what to do next? Or just want a quick trip to your happy place.


## Installation and Requirements

To install the packages, run 

``` pip install -r requirements.txt ```

The system will work only on a GPU. It is preferred to have Cuda11.3 and above and ```torch>=2.4.0```.

## Usage

After the installation, you can play around with the notebook for text input and voice output.

For a only voice assisted system run the command 

```python immersivehack_script.py```

This project uses Meta's LLaMA3.2 3B-Instruct model, and one can use different models to test and verify the working 
with different models from HuggingFace.

