README FILE
----------------------------------------------------------------
Project:
The main goal is to build a deep learning model that understands 
the sign language in real time and can be used to transform it into
voice.

STEPS TO RUN THE PROJECT
----------------------------------------------------------------
Create a Virtual environment with the requirements. You can run the app.py with my mode. But this is the process to do
your own model.

Step 1: Create the folder for the data.

 ---> folders_for_collection.py

Step 2: Collect the data.
We use our own data to train the model so we use our webcam to
get data from all the different actions.

 ---> collect_data.py

Step 3: Preprocess the data.

 ---> preprocess_data.py

Step 4: Build the model and train the model.

 ---> build_train.py

Step 5: Run the app with all the logic.

  ---> app.py


DEPENDENCIES
-----------------------------------------------------------------
tensorflow: Create and train deep learning models.
opencv-python: Access webcam and extract key points.
mediapipe: Real-time tracking of hands, faces, poses.
sklearn: Evaluation metrics, training and testing split.
gtts: Google Text-to-Speech

(Check out the requirements.txt to find more information).


TECH EXPLANATION
----------------------------------------------------------------
To train my deep learning model I am using LSTM (Long Short Term Memory).

LSTMs consist of memory cells that can store information, and gates that
control the flow of data into, out of, and within the cell. These gates
help the network decide what information to keep or forget, enabling it
to learn patterns over long sequences, which is essential when interpretin
sign language.

I feed the model whit my own data, then I train it with 2000 Epocs and
finally I made the logic for the app.

