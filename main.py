from vocode.streaming.utils import get_chunk_size_per_second
from vocode.streaming.synthesizer import *
from vocode.streaming.output_device.speaker_output import SpeakerOutput
from vocode.streaming.output_device.base_output_device import BaseOutputDevice
import vocode
from vocode.streaming.transcriber import *
from vocode.streaming.transcriber.base_transcriber import BaseTranscriber, Transcription
from vocode.streaming.models.transcriber import (
    DeepgramTranscriberConfig,
    PunctuationEndpointingConfig,
)
from vocode.streaming.input_device.microphone_input import MicrophoneInput
import base64
from vocode.streaming.models.synthesizer import (
    ElevenLabsSynthesizerConfig,
)
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.agent.bot_sentiment_analyser import BotSentiment
import aiohttp
from typing import Optional
import argparse
import openai
import os
import time
from spot_controller import SpotController
import asyncio
from dotenv import load_dotenv
ROBOT_IP = "192.168.50.3"  # os.environ['ROBOT_IP']
SPOT_USERNAME = "admin"  # os.environ['SPOT_USERNAME']
SPOT_PASSWORD = "2zqa8dgw7lor"  # os.environ['SPOT_PASSWORD']


# from playground.streaming.tracing_utils import make_parser_and_maybe_trace
# these can also be set as environment variables
vocode.setenv(
    OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY'),
    DEEPGRAM_API_KEY=os.environ.get('DEEPGRAM_API_KEY'),
    ELEVEN_LABS_API_KEY=os.environ.get('ELEVEN_LABS_API_KEY')
)


openai.api_key = "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"


if __name__ == "__main__":

    speaker_output = SpeakerOutput.from_default_device()
    synthesizer = ElevenLabsSynthesizer(
        ElevenLabsSynthesizerConfig.from_output_device(speaker_output)
    )
    load_dotenv()

    async def print_output(transcriber: BaseTranscriber, spot):
        while True:
            transcription: Transcription = await transcriber.output_queue.get()
            print(transcription)

            # If the text is in the message, then perform the action
            if "spot" in transcription.message.lower():
                print("Spot is listening")
                # Rotate spot
                print(transcription.message)

                _ = await speak(
                    synthesizer=synthesizer,
                    output_device=speaker_output,
                    message=BaseMessage(text="I am analyzing my surroundings"),),

                # Rotate Spot

                # Move head to specified positions with intermediate time.sleep
                spot.move_head_in_points(yaws=[0.2, 0],
                                         pitches=[0.3, 0],
                                         rolls=[0.4, 0],
                                         sleep_after_point_reached=1)
                time.sleep(3)

                # Make Spot to move by goal_x meters forward and goal_y meters left
                spot.move_to_goal(goal_x=0.5, goal_y=0)
                time.sleep(3)

                # Control Spot by velocity in m/s (or in rad/s for rotation)
                spot.move_by_velocity_control(
                    v_x=-0, v_y=0, v_rot=0.3, cmd_duration=15)
                time.sleep(3)

                # Use GPT-4V to take the image and analyze it

                image_path = "photo.jpg"

                def encode_image(image_path):
                    with open(image_path, "rb") as image_file:
                        return base64.b64encode(image_file.read()).decode('utf-8')
                # Getting the base64 string
                base64_image = encode_image(image_path)
                response = openai.ChatCompletion.create(
                    model="gpt-4-vision-preview",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "What's in this image?"},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}",
                                    },
                                },
                            ],
                        }
                    ],
                    max_tokens=300,
                )

                # Speak the image
                text = response.choices[0]['message']['content']
                print(text)
                _ = await speak(
                    synthesizer=synthesizer,
                    output_device=speaker_output,
                    message=BaseMessage(text=text[:500]),),
            print(transcription.message)

    # make_parser_and_maybe_trace()

    seconds_per_chunk = 1

    async def speak(
        synthesizer: BaseSynthesizer,
        output_device: BaseOutputDevice,
        message: BaseMessage,
        bot_sentiment: Optional[BotSentiment] = None,
    ):
        message_sent = message.text
        cut_off = False
        chunk_size = seconds_per_chunk * get_chunk_size_per_second(
            synthesizer.get_synthesizer_config().audio_encoding,
            synthesizer.get_synthesizer_config().sampling_rate,
        )
        # ClientSession needs to be created within the async task
        synthesizer.aiohttp_session = aiohttp.ClientSession()
        synthesis_result = await synthesizer.create_speech(
            message=message,
            chunk_size=chunk_size,
            bot_sentiment=bot_sentiment,
        )
        chunk_idx = 0
        async for chunk_result in synthesis_result.chunk_generator:
            try:
                start_time = time.time()
                speech_length_seconds = seconds_per_chunk * (
                    len(chunk_result.chunk) / chunk_size
                )
                output_device.consume_nonblocking(chunk_result.chunk)
                end_time = time.time()
                await asyncio.sleep(
                    max(
                        speech_length_seconds - (end_time - start_time),
                        0,
                    )
                )
                print(
                    "Sent chunk {} with size {}".format(
                        chunk_idx, len(chunk_result.chunk)
                    )
                )
                chunk_idx += 1
            except asyncio.CancelledError:
                seconds = chunk_idx * seconds_per_chunk
                print(
                    "Interrupted, stopping text to speech after {} chunks".format(
                        chunk_idx
                    )
                )
                message_sent = f"{synthesis_result.get_message_up_to(seconds)}-"
                cut_off = True
                break

        return message_sent, cut_off

    async def main():

        # # Capture image
        # import cv2
        # camera_capture = cv2.VideoCapture(0)
        # rv, image = camera_capture.read()
        # print(f"Image Dimensions: {image.shape}")
        # camera_capture.release()

        # Use wrapper in context manager to lease control, turn on E-Stop, power on the robot and stand up at start
        # and to return lease + sit down at the end
        with SpotController(username=SPOT_USERNAME, password=SPOT_PASSWORD, robot_ip=ROBOT_IP) as spot:

            time.sleep(2)

            microphone_input = MicrophoneInput.from_default_device()

            # replace with the transcriber you want to test
            transcriber = DeepgramTranscriber(
                DeepgramTranscriberConfig.from_input_device(
                    microphone_input, endpointing_config=PunctuationEndpointingConfig()
                )
            )
            transcriber.start()
            asyncio.create_task(print_output(transcriber, spot))
            print("Start speaking...press Ctrl+C to end. ")
            while True:
                chunk = await microphone_input.get_audio()
                transcriber.send_audio(chunk)

    asyncio.run(main())
