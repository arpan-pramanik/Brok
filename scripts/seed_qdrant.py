import os
import uuid
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = "msmarco_xi"

PASSAGES = [
    # AI & Computer Science
    "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to intelligence of humans and other animals. Example tasks in which this is done include speech recognition, computer vision, translation between natural languages, as well as other mappings of inputs.",
    "Machine learning (ML) is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalize to unseen data, and thus perform tasks without explicit instructions.",
    "Deep learning is a subset of machine learning based on artificial neural networks with representation learning. The adjective 'deep' refers to the use of multiple layers in the network.",
    "Natural Language Processing (NLP) is an interdisciplinary subfield of computer science and linguistics. It is primarily concerned with giving computers the ability to support and manipulate human language.",
    "Computer vision is an interdisciplinary scientific field that deals with how computers can be made to gain high-level understanding from digital images or videos.",
    "Reinforcement learning (RL) is an area of machine learning concerned with how intelligent agents ought to take actions in an environment in order to maximize the notion of cumulative reward.",
    "The Turing test, originally called the imitation game by Alan Turing in 1950, is a test of a machine's ability to exhibit intelligent behavior equivalent to, or indistinguishable from, that of a human.",
    "Supervised learning is a machine learning paradigm where the algorithm learns a model from labeled training data containing both input variables and target outputs.",
    "Unsupervised learning is a type of machine learning algorithm used to draw inferences from datasets consisting of input data without labeled responses.",
    "A neural network is a computation model inspired by the structure of biological neural networks. It consists of interconnected nodes or artificial neurons organized in layers.",
    "Convolutional Neural Networks (CNNs) are a class of deep neural networks, most commonly applied to analyzing visual imagery in computer vision applications.",
    "Recurrent Neural Networks (RNNs) are a class of artificial neural networks where connections between nodes can form a directed graph along a temporal sequence, making them suitable for sequential data like speech and text.",
    "Transformers are a deep learning model architecture introduced in 2017 that relies entirely on self-attention mechanisms to compute representations of its input and output without using sequence-aligned RNNs or convolution.",
    "Overfitting occurs in machine learning when a statistical model begins to memorize noise and details in the training data rather than learning the underlying generalization rule.",
    "Backpropagation is a widely used algorithm in training artificial neural networks to calculate the gradient of the loss function with respect to each weight by the chain rule.",

    # History & Manhattan Project
    "The Manhattan Project was a research and development undertaking during World War II that produced the first nuclear weapons. It was led by the United States with the support of the United Kingdom and Canada.",
    "Major General Leslie Groves of the U.S. Army Corps of Engineers directed the Manhattan Project from 1942 to 1946.",
    "Nuclear physicist J. Robert Oppenheimer was the director of the Los Alamos Laboratory that designed the actual nuclear weapons during the Manhattan Project.",
    "The first nuclear weapon test, code-named Trinity, was conducted by the United States Army at 5:29 a.m. on July 16, 1945, in the Jornada del Muerto desert near Socorro, New Mexico.",
    "Enrico Fermi achieved the first controlled, self-sustaining nuclear chain reaction in the Chicago Pile-1 reactor on December 2, 1942.",
    "The B-29 Superfortress named Enola Gay dropped the first atomic bomb, Little Boy, on the Japanese city of Hiroshima on August 6, 1945.",
    "The second atomic bomb, Fat Man, was dropped on Nagasaki on August 9, 1945 by the B-29 bomber Bockscar.",
    "The Treaty of Versailles was signed on June 28, 1919, officially ending World War I.",
    "The Industrial Revolution began in Great Britain in the mid-18th century, transitioning manufacturing processes to new machinery, steam power, and mechanized factory systems.",

    # Biology & Photosynthesis
    "Photosynthesis is a system of biological processes by which photosynthetic organisms, such as most plants, algae, and cyanobacteria, convert solar energy into chemical energy.",
    "Photosynthesis converts light energy, water, and carbon dioxide into oxygen and energy-rich organic compounds, specifically glucose.",
    "Chlorophyll is the primary green pigment found in plants and cyanobacteria that absorbs light energy required for photosynthesis.",
    "Cellular respiration is a set of metabolic reactions and processes that take place in the cells of organisms to convert biochemical energy from nutrients into adenosine triphosphate (ATP).",
    "DNA (deoxyribonucleic acid) is a molecule composed of two polynucleotide chains that coil around each other to form a double helix carrying genetic instructions for the development, functioning, growth, and reproduction of all known organisms.",
    "RNA (ribonucleic acid) is a polymeric molecule essential in various biological roles in coding, decoding, regulation, and expression of genes.",
    "Mitochondria are membrane-bound cell organelles that generate most of the chemical energy needed to power the cell's biochemical reactions, earned as ATP.",
    "Enzymes are proteins that act as biological catalysts by accelerating chemical reactions in cells without being consumed in the process.",

    # Physics & Quantum Mechanics
    "Quantum computing is a rapidly-emerging technology that harnesses the laws of quantum mechanics to solve problems too complex for classical computers.",
    "Qubits (quantum bits) are the basic unit of quantum information, capable of existing in superpositions of 0 and 1 simultaneously.",
    "Einstein's theory of special relativity states that the laws of physics are the same for all non-accelerating observers and that the speed of light in a vacuum is independent of the motion of all observers.",
    "The equivalence principle of general relativity postulates that the gravitational force experienced locally while standing on a massive body is identical to the pseudo-force experienced by an observer in a non-inertial frame.",
    "Thermodynamics is a branch of physics that deals with heat, work, temperature, and their relation to energy, entropy, radiation, and physical properties of matter.",
    "The First Law of Thermodynamics states that energy can neither be created nor destroyed, only transformed from one form to another.",
    "The Second Law of Thermodynamics states that the total entropy of an isolated system always increases over time.",
    "Electromagnetic radiation consists of waves of the electromagnetic field, propagating through space, carrying electromagnetic radiant energy including gamma rays, X-rays, ultraviolet, visible light, infrared, microwaves, and radio waves.",

    # Mathematics & Space
    "Calculus is the mathematical study of continuous change, dividing into differential calculus regarding rates of change and slopes of curves, and integral calculus regarding accumulation of quantities.",
    "The Pythagorean theorem states that in a right-angled triangle, the square of the length of the hypotenuse is equal to the sum of the squares of the lengths of the other two sides: a^2 + b^2 = c^2.",
    "Prime numbers are natural numbers greater than 1 that have no positive divisors other than 1 and themselves.",
    "The solar system consists of the Sun and the objects that orbit it, including eight major planets: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune.",
    "Light-year is a unit of length used to express astronomical distances and is equivalent to about 9.46 trillion kilometers or 5.88 trillion miles.",
]

def main():
    print(f"Connecting to Qdrant at {QDRANT_URL}...")
    client = QdrantClient(url=QDRANT_URL)

    if client.collection_exists(COLLECTION):
        print(f"Deleting existing collection {COLLECTION}...")
        client.delete_collection(COLLECTION)

    print(f"Creating collection {COLLECTION} with 384-dim COSINE vector config...")
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

    print("Initializing FastEmbed BGESmallENV15 model...")
    embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    print(f"Embedding {len(PASSAGES)} in-domain passages...")
    embeddings = list(embed_model.embed(PASSAGES))

    points = []
    for i, (text, emb) in enumerate(zip(PASSAGES, embeddings)):
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=emb.tolist(),
            payload={
                "text": text,
                "source_doc": f"msmarco_doc_{i+1}",
                "chunk_index": i,
                "language": "en"
            }
        ))

    print(f"Upserting {len(points)} points into Qdrant...")
    client.upsert(collection_name=COLLECTION, points=points)

    print(f"Successfully seeded {len(points)} passages into Qdrant collection '{COLLECTION}'!")

if __name__ == "__main__":
    main()
