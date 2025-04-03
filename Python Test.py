import pandas as pd
import random
from faker import Faker

# Initialize Faker
fake = Faker()

# Define hackathon domains
domains = ["AI & ML", "Blockchain", "Cybersecurity", "IoT", "Web Development"]

# Generate dataset
data = []
for _ in range(350):
    participant = {
        "Participant_ID": fake.unique.random_int(min=1000, max=9999),
        "Name": fake.name(),
        "Email": fake.email(),
        "Age": random.randint(18, 40),
        "Gender": random.choice(["Male", "Female", "Other"]),
        "Day": random.choice([1, 2, 3]),
        "Hackathon_Domain": random.choice(domains),
        "Score": random.randint(50, 100),  # Random score
        "Feedback": fake.sentence(),
        "City": fake.city(),
    }
    data.append(participant)

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("hackathon_participants.csv", index=False)

print("Dataset generated successfully and saved as 'hackathon_participants.csv'")


