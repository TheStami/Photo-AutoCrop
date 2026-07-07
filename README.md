# AutoCrop

AutoCrop is a simple, AI-powered tool that automatically detects the main subject in your photos and sets the optimal crop area. With its interactive editor, you can preview the results in real-time and easily fine-tune the bounding box to get the perfect crop.

## How to run

1. Ensure you have Python 3 installed on your system.
2. Create and activate a virtual environment (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the graphical user interface:
   ```bash
   python3 gui.py
   ```

## How to use

1. Launch the application.
2. Load an image using the file selection dialog.
3. The AI will process the image automatically and detect the main subject, drawing a bounding box around it.
4. You can adjust the bounding box manually by dragging its corners or edges.
   - Hold `Shift` while dragging for axis-aligned rectangle constraints.
5. Use the arrow keys to rotate the image in 90-degree increments.
6. Use `Tab` and `Shift+Tab` to quickly navigate between loaded images.
7. The real-time preview window displays the final cropped result.
8. Once you are satisfied with the result, save the cropped image.
