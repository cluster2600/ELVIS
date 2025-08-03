import genanki

model = genanki.Model(
    1607392319,
    'NVIDIA Associate Flashcard Model',
    fields=[{'name': 'Question'}, {'name': 'Answer'}],
    templates=[{
        'name': 'Card 1',
        'qfmt': '{{Question}}',
        'afmt': '{{FrontSide}}<hr id="answer">{{Answer}}',
    }]
)

deck = genanki.Deck(2059400110, 'NVIDIA Associate: AI Infrastructure & Operations')

cards = [
    ('What is CUDA?', 'NVIDIA\'s parallel computing platform and API for GPU programming.'),
    ('What does DCGM stand for and do?', 'Data Center GPU Manager; it monitors GPU health and telemetry.'),
    ('What is NGC?', 'NVIDIA GPU Cloud — a registry of optimized containers and models for AI/HPC.'),
    ('What does NCCL optimize?', 'Inter-GPU communication for distributed deep learning training.'),
    ('What is MIG?', 'Multi-Instance GPU — partitions a GPU into isolated compute instances.'),
    ('How do you see ECC errors and power draw?', '`nvidia-smi -q`'),
    ('What is DeepOps used for?', 'Automated deployment of GPU infrastructure including drivers and Slurm.'),
    ('What is the purpose of the NVIDIA GPU Operator?', 'Deploys GPU drivers and runtimes in Kubernetes environments.'),
    ('What is vGPU?', 'Virtual GPU — allows multiple VMs to share a physical GPU.'),
    ('Which command sets a GPU power limit?', '`nvidia-smi --power-limit=`'),
]

for q, a in cards:
    deck.add_note(genanki.Note(model=model, fields=[q, a]))

genanki.Package(deck).write_to_file('NVIDIA_Associate_Deck.apkg')
