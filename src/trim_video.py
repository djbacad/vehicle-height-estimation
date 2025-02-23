import argparse
from moviepy import VideoFileClip
import os

def mmss_to_seconds(time_str):
    """Convert a time string 'MM:SS' to total seconds."""
    minutes, seconds = map(int, time_str.split(':'))
    return minutes * 60 + seconds

def main():
    parser = argparse.ArgumentParser(description="Trim a video file to a specified time range.")
    parser.add_argument('video_filename', help="Name of the input video file located in 'data/video_full/'.")
    parser.add_argument('start_time', help="Start time in 'MM:SS' format.")
    parser.add_argument('end_time', help="End time in 'MM:SS' format.")
    args = parser.parse_args()

    input_dir = 'data/for_trim/'
    output_dir = 'data/videos/'

    # Construct full input path
    input_path = os.path.join(input_dir, args.video_filename)

    # Validate input file existence
    if not os.path.isfile(input_path):
        print(f"Error: The file '{input_path}' does not exist.")
        return

    # Construct output filename and path
    output_filename = f"{os.path.splitext(args.video_filename)[0]}_trimmed.mp4"
    output_path = os.path.join(output_dir, output_filename)

    # Convert 'MM:SS' to seconds
    start_time = mmss_to_seconds(args.start_time)
    end_time = mmss_to_seconds(args.end_time)

    # Load the video file
    subclip = VideoFileClip(input_path).subclipped(start_time, end_time)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Write the result to a file
    subclip.write_videofile(output_path, codec="libx264")

if __name__ == "__main__":
    main()