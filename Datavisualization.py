import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import numpy as np
from PIL import Image

# Load dataset
df = pd.read_csv("hackathon_participants.csv")

# Streamlit page setup
st.set_page_config(page_title="Hackathon Dashboard", layout="wide")

st.title("Hackathon Event Dashboard")
st.sidebar.header("Filters")

# Sidebar filters
selected_domain = st.sidebar.multiselect("Select Hackathon Domain", df["Hackathon_Domain"].unique(), default=df["Hackathon_Domain"].unique())
selected_state = st.sidebar.multiselect("Select City/State", df["City"].unique(), default=df["City"].unique())

# Apply filters
filtered_df = df[(df["Hackathon_Domain"].isin(selected_domain)) & (df["City"].isin(selected_state))]

st.write("### Overview of Participants")
st.dataframe(filtered_df.head(10))

# Function to plot bar chart
def plot_bar_chart(data, x_col, title):
    plt.figure(figsize=(8, 4))
    sns.countplot(data=data, x=x_col, palette="viridis")
    plt.title(title)
    plt.xticks(rotation=45)
    st.pyplot(plt)

# Visualization
st.write("## Data Visualizations")

col1, col2 = st.columns(2)

with col1:
    plot_bar_chart(filtered_df, "Hackathon_Domain", "Participants per Domain")

with col2:
    plot_bar_chart(filtered_df, "Day", "Participation per Day")

col3, col4 = st.columns(2)

with col3:
    plot_bar_chart(filtered_df, "City", "Participants per City (State-wise)")

with col4:
    plot_bar_chart(filtered_df, "Gender", "Gender Distribution")

# Score Distribution
st.write("### Score Distribution of Participants")
plt.figure(figsize=(8, 4))
sns.histplot(filtered_df["Score"], bins=10, kde=True, color="blue")
plt.title("Score Distribution")
st.pyplot(plt)

st.write("### Summary Statistics")
st.write(filtered_df.describe())

st.write("### Thank you for using the Hackathon Dashboard!")

''' Image Processing Module '''

st.write("## Image Processing Module")

# Custom Image Processing Component
st.write("## Upload an Image for Processing")

uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Original Image", use_column_width=True)

    # Convert image to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Processing options
    option = st.radio("Select Processing", ["Grayscale", "Edge Detection", "Blurring", "Sharpening"])

    if option == "Grayscale":
        processed_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        st.image(processed_image, caption="Grayscale Image", use_column_width=True, channels="GRAY")

    elif option == "Edge Detection":
        # Convert to grayscale first
        gray_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian Blur to remove noise
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        # Apply Canny Edge Detection
        processed_image = cv2.Canny(blurred_image, 50, 150)
        st.image(processed_image, caption="Edge Detected Image", use_column_width=True, channels="GRAY")

    elif option == "Blurring":
        # Apply Gaussian Blur
        processed_image = cv2.GaussianBlur(image_cv, (15, 15), 0)
        st.image(processed_image, caption="Blurred Image", use_column_width=True, channels="BGR")

    elif option == "Sharpening":
        # Define sharpening kernel
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        processed_image = cv2.filter2D(image_cv, -1, kernel)
        st.image(processed_image, caption="Sharpened Image", use_column_width=True, channels="BGR")


