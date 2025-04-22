# Face Anti-Spoofing Detection App

This application uses deep learning to detect face spoofing attacks in real-time using your webcam or uploaded images. It can distinguish between real faces (live person) and fake faces (photos, videos, masks, etc.).

## Deployment on Streamlit Cloud

### Prerequisites

- A [Streamlit Cloud](https://streamlit.io/cloud) account
- A GitHub repository with this codebase

### Deployment Steps

1. **Fork/Upload this repository to your GitHub account**

2. **Ensure the model file is properly included**
   - The model file should be in the `model_checkpoints` directory
   - The model path in the code is set to `model_checkpoints/model.08-1.00.keras`

3. **Deploy on Streamlit Cloud**
   - Go to [Streamlit Cloud](https://streamlit.io/cloud)
   - Click on "New app"
   - Select your GitHub repository
   - In the main file path field, enter: `streamlit_app.py`
   - Click "Deploy"

4. **Advanced Settings (Optional)**
   - You can adjust memory limits, Python version, etc., in the advanced settings
   - For this application, the default settings should work fine
   - No need to add secrets or environment variables

## Local Development

To run the application locally:

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the Streamlit app:
   ```
   streamlit run streamlit_app.py
   ```

## Project Structure

- `streamlit_app.py`: Main application file for Streamlit Cloud deployment
- `Testing/`: Directory containing original application files
- `model_checkpoints/`: Directory containing the trained model
- `requirements.txt`: List of Python dependencies
- `.streamlit/config.toml`: Streamlit configuration settings

## Important Notes

1. **CPU Compatibility**: This application is configured to run on CPU, as Streamlit Cloud does not support GPU acceleration. The TensorFlow model has been configured to use the CPU.

2. **Webcam Access**: The app uses WebRTC to access your camera, which requires user permission in the browser.

3. **Model Size**: The face anti-spoofing model is approximately 539MB. Make sure your Streamlit Cloud account has enough storage to accommodate this.

4. **Processing Speed**: Without GPU acceleration, face detection and anti-spoofing prediction may be slightly slower than in a local environment with GPU support.

## Troubleshooting

- **Camera not working**: Make sure to grant camera permission in your browser settings
- **Slow performance**: The CPU-based processing might be slower than GPU, this is expected
- **Model not loading**: Check if the model path is correct and the model file is properly uploaded
- **"No face detected"**: Make sure your face is clearly visible in the webcam view or uploaded image

For additional help, please file an issue in the GitHub repository. 