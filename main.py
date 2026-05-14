import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from datetime import datetime
import pandas as pd
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow import keras
import smtplib
from email.message import EmailMessage

# Load pre-trained MobileNetV2 for species classification
model = keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=True, weights='imagenet')


# Function to classify tree species and filter out jellyfish
def classify_species(image):
    img = cv2.resize(image, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0) / 255.0  # Normalize pixel values
    species_predictions = model.predict(img)
    decoded_predictions = keras.applications.mobilenet_v2.decode_predictions(species_predictions)

    # Exclude jellyfish from results
    filtered_predictions = [(label, confidence) for _, label, confidence in decoded_predictions[0][:3] if
                            "jellyfish" not in label.lower()]

    return filtered_predictions


# Function to generate a simulated "after" deforestation image
def simulate_deforestation(image_path):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    image[mask > 0] = (255, 0, 0)  # Change green areas to blue
    simulated_path = "simulated_after.jpg"
    cv2.imwrite(simulated_path, image)
    return simulated_path


# Function to count trees and classify species
def analyze_forest(image_path, min_tree_area, threshold_value):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tree_count = sum(1 for contour in contours if cv2.contourArea(contour) > min_tree_area)
    species_predictions = classify_species(image) if tree_count > 0 else []

    return tree_count, species_predictions


# Function to send email with attached report
def send_email(file_path):
    sender_email = "deforestationmonitoringsystem@gmail.com"
    sender_password = "fzna jerl xcwo zjrm"  # Use an App Password

    recipient_email = simpledialog.askstring("Recipient Email", "Enter recipient's email:")
    if not recipient_email:
        messagebox.showerror("Error", "Recipient email is required.")
        return

    msg = EmailMessage()
    msg["Subject"] = "Deforestation Alert: Immediate Action Required!"
    msg["From"] = sender_email
    msg["To"] = recipient_email

    body = "Dear Authorities,\n\nA deforestation event has been detected. Please find the attached report for necessary actions.\n\nBest Regards,\nDeforestation Monitoring System"
    msg.set_content(body)

    with open(file_path, "rb") as f:
        file_data = f.read()
        file_name = os.path.basename(file_path)
        msg.add_attachment(file_data, maintype="application", subtype="octet-stream", filename=file_name)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)
        messagebox.showinfo("Success", "Deforestation report sent successfully.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to send email: {e}")


class DeforestationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Deforestation & Tree Species Prediction")

        self.image_path = None
        self.forest_name = None
        self.min_tree_area = 300
        self.threshold_value = 150
        self.results = []

        self.create_gui()

    def create_gui(self):
        open_button = tk.Button(self.root, text="Open Forest Image", command=self.open_image)
        open_button.pack()

        process_button = tk.Button(self.root, text="Simulate & Predict Deforestation", command=self.process_image)
        process_button.pack()

        export_button = tk.Button(self.root, text="Export & Send Report", command=self.export_data)
        export_button.pack()

        self.image_label = tk.Label(self.root)
        self.image_label.pack()

        self.results_text = tk.Text(self.root, height=15, width=70)
        self.results_text.pack()

    def open_image(self):
        file_path = filedialog.askopenfilename(title="Select a Forest Image",
                                               filetypes=[("Images", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.image_path = file_path
            self.forest_name = simpledialog.askstring("Forest Name", "Enter the forest area name:")
            if not self.forest_name:
                messagebox.showerror("Error", "Forest name is required.")
                return
            self.show_image(self.image_path)

    def process_image(self):
        if not self.image_path:
            self.results_text.insert(tk.END, "Error: Select an image first.\n")
            return

        before_image = self.image_path
        after_image = simulate_deforestation(before_image)

        before_count, before_species = analyze_forest(before_image, self.min_tree_area, self.threshold_value)
        after_count, after_species = analyze_forest(after_image, self.min_tree_area, self.threshold_value)

        tree_loss = before_count - after_count
        tree_loss_percentage = (tree_loss / before_count * 100) if before_count > 0 else 0
        deforestation_status = "deforestation not happening"

        if tree_loss_percentage > 5:
            deforestation_status = f"Deforestation Detected: {tree_loss} trees lost ({tree_loss_percentage:.2f}% loss)."

        result = {
            "Forest Name": self.forest_name,
            "Tree Loss": tree_loss,
            "Tree Loss (%)": f"{tree_loss_percentage:.2f}%",
            "Before Species": before_species,
            "After Species": after_species,
            "Deforestation Status": deforestation_status,
            "Date": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        self.results.append(result)

        self.display_results(result)
        self.show_image(after_image)

    def display_results(self, result):
        self.results_text.delete(1.0, tk.END)

        self.results_text.insert(tk.END, f"Forest Name: {result['Forest Name']}\n")
        self.results_text.insert(tk.END, f"Tree Loss: {result['Tree Loss']} ({result['Tree Loss (%)']})\n")
        self.results_text.insert(tk.END, f"Status: {result['Deforestation Status']}\n\n")

        self.results_text.insert(tk.END, "Species Before Deforestation:\n")
        for species, confidence in result['Before Species']:
            self.results_text.insert(tk.END, f"{species}: {confidence:.2%}\n")

        self.results_text.insert(tk.END, "\nSpecies After Deforestation:\n")
        for species, confidence in result['After Species']:
            self.results_text.insert(tk.END, f"{species}: {confidence:.2%}\n")

        self.results_text.insert(tk.END, f"\nDate: {result['Date']}\n\n")

    def show_image(self, image_path):
        image = Image.open(image_path)
        image = image.resize((300, 300))
        image = ImageTk.PhotoImage(image)
        self.image_label.config(image=image)
        self.image_label.image = image

    def export_data(self):
        if not self.results:
            self.results_text.insert(tk.END, "No results to export.\n")
            return

        df = pd.DataFrame(self.results)
        export_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                   filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")])
        if export_path:
            if export_path.endswith(".xlsx"):
                df.to_excel(export_path, index=False)
            else:
                df.to_csv(export_path, index=False)
            send_email(export_path)


if __name__ == "__main__":
    root = tk.Tk()
    app = DeforestationApp(root)
    root.mainloop()