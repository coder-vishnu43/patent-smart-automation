import pandas as pd
import random
from faker import Faker

fake = Faker()
trl_levels = ['Low', 'Medium', 'High']
domains = ['Quantum Computing', 'AI', 'Robotics', 'Biotech']
keywords_dict = {
    'Quantum Computing': ['Quantum Processor', 'Superconducting Qubits', 'Trapped Ion Qubits', 'Quantum Algorithm'],
    'AI': ['Deep Learning', 'Computer Vision', 'NLP', 'Reinforcement Learning'],
    'Robotics': ['Autonomous Robots', 'Swarm Robotics', 'Industrial Robotics', 'Humanoid Robots'],
    'Biotech': ['Gene Editing', 'Synthetic Biology', 'CRISPR', 'Bioinformatics']
}

data = []
for i in range(60):
    domain = random.choice(domains)
    keywords = random.sample(keywords_dict[domain], k=2)
    title = f"{domain} Research {i+1}"
    year = random.randint(2015, 2025)
    org = fake.company()
    trl = random.choices(trl_levels, weights=[0.4,0.4,0.2])[0]
    data.append([title, domain, year, org, ', '.join(keywords), trl])

df = pd.DataFrame(data, columns=['Title','Domain','Year','Organization','Keywords','TRL'])
df.to_csv('data/multi_domain_demo_data.csv', index=False)
print("Multi-domain sample data generated successfully!")
