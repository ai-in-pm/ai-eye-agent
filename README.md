# AI Eye Agent

An intelligent eye-tracking agent that learns from your gaze patterns to control your computer cursor and execute commands.

## First iteration of running the AI Eye Agent



The AI Eye Agent completed the following:

  - Open your webcam
  - Show a fullscreen black window with a green dot
  - Move the dot to 9 different positions
  - Train the model on your eye movements
  - Start controlling the cursor based on your gaze

  - The AI Eye Agent was in full control of my cursor, did not allow me to physically control it, as shown in this video.  I had to conduct hard restart to my computer even though I was on a Virtual environment.
    
    https://youtu.be/aotrpSxUZC4

## Features

- Eye tracking-based cursor control
- Self-learning capability to improve accuracy over time
- System command execution (open browser, documents, etc.)
- Integration with OpenAI's GPT-4 for natural language understanding

## Prerequisites

- Python 3.7 or higher
- Webcam
- Windows operating system
- OpenAI API key
- PsychoPy
- OpenCV

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/AI_Eye_Agent.git
   cd AI_Eye_Agent
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

1. Run the AI Eye Agent:
   ```bash
   python ai_eye_agent.py
   ```

2. The first time you run the agent, it will:
   - Calibrate the eye tracker
   - Collect training data for 60 seconds (follow cursor with your eyes)
   - Train the AI model
   - Start controlling your cursor based on eye movements

3. To stop the agent, press Ctrl+C

## How it Works

1. **Eye Tracking**: Uses PyGaze to track your eye movements
2. **Machine Learning**: Trains a neural network to map eye positions to cursor positions
3. **AI Integration**: Uses GPT-4 to understand and execute system commands
4. **Continuous Learning**: Improves accuracy over time through usage

## Configuration

The agent requires the following environment variables in your `.env` file:
- `OPENAI_API_KEY`: Your OpenAI API key for GPT-4 integration (Required)

Make sure to keep your `.env` file secure and never commit it to version control.

## Notes

- The initial training period is crucial for accuracy
- Keep your head relatively stable during use
- The agent works best on a single monitor setup initially
- Performance may vary based on lighting conditions and eye tracker quality

## The development of this GitHub Repository was inspired by Apple's Iphone IOS 18.3 Developer Beta Update - Eye Tracking feauture
![Apple IOS 18_3 Developer Beta Update I](https://github.com/user-attachments/assets/020f153a-2184-422f-8513-0205c19190ef)
![Apple IOS 18_3 Developer Beta Update II](https://github.com/user-attachments/assets/02b3ad83-7893-46a8-830b-191e38fad8a4)

