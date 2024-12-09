import streamlit as st
import cv2
import os
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import base64

def main():
    st.title("ProPlay Analysis App")
    st.sidebar.title("Upload Video")

    # Streamlit field for OpenAI API key
    api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

    if not api_key:
        st.warning("Please enter your OpenAI API Key to proceed.")
        return

    # Set up the LangChain ChatOpenAI model dynamically with the provided API key
    llm = ChatOpenAI(model="gpt-4o", api_key=api_key)

    # ChatPromptTemplate setup
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a sports coach. Analyze the frame from a sports video provided in the description below."),
            (
                "human",
                [
                    {"type": "text", "text": "Provide actionable suggestions for the player to improve their game:"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,""{image}",
                            "detail": "low",
                        },
                    },
                ],
            ),
        ]
    )

    chain = prompt | llm

    def extract_frames(video_path, output_dir, frame_rate=1):
        """Extract frames from a video at a given frame rate."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        cap = cv2.VideoCapture(video_path)
        count = 0
        success = True
        while success:
            success, frame = cap.read()
            if success and count % frame_rate == 0:
                frame_path = os.path.join(output_dir, f"frame_{count}.jpg")
                cv2.imwrite(frame_path, frame)
            count += 1
        cap.release()

    def frame_to_base64(frame_path):
        """Convert an image frame to a base64-encoded string."""
        with open(frame_path, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode("utf-8")
        return encoded_string

    def analyze_frame_with_langchain(frame_path):
        frame_data = frame_to_base64(frame_path)
        response = chain.invoke({"image": frame_data})
        return response

    # File uploader for video
    uploaded_video = st.sidebar.file_uploader("Upload a sports video", type=["mp4", "avi", "mov"])

    if uploaded_video:
        # Save uploaded video temporarily
        video_path = f"temp_{uploaded_video.name}"
        with open(video_path, "wb") as f:
            f.write(uploaded_video.read())

        st.video(video_path)

        if st.button("Analyze Video"):
            st.write("Extracting frames from the video...")
            output_dir = "frames"
            extract_frames(video_path, output_dir)

            st.write("Analyzing frames...")
            suggestions = []
            count = 0
            for frame_file in sorted(os.listdir(output_dir)):
                count += 1
                if count < 2:
                    frame_path = os.path.join(output_dir, frame_file)
                    suggestion = analyze_frame_with_langchain(frame_path)
                    suggestions.append(f"Frame {frame_file}: {suggestion}")

            st.write("Analysis Completed!")
            st.write("Suggestions for improvement:")
            for suggestion in suggestions:
                final_suggestion = llm.invoke("summarize in 5 points " + suggestion)
                st.write(final_suggestion.content)

        # Clean up
        if os.path.exists(video_path):
            os.remove(video_path)

if __name__ == "__main__":
    main()
