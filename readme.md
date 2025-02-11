
### Installation Command

First, ensure you have the necessary packages installed. You can do this using pip. Run the following command in your terminal:

```
pip install Flask tensorflow numpy
```
### Running the Application

Once the required packages are installed, you can run the application with the following command:

```
gunicorn app:app --bind 0.0.0.0:10000
```

### Making a Prediction


To make a prediction using the application, you can use the following `curl` command. Replace `test/try.png` with the path to your image file:

```
curl -X POST -F "file=@test/test.jpeg" http://localhost:10000/predict/pneumonia

```