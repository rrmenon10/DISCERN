set -exu

# Dataset name from the first argument
DATASET_NAME=$1

NUM_TRAIN_EXAMPLES=$(python - << END
# Dictionary mapping datasets to number of examples
dataset_config = {
    "data/trec": 1500,
    "data/ag_news": 2000,
    "data/covid": 4000,
}

# Get dataset name from bash
dataset_name = "$DATASET_NAME"

# Default number of examples if dataset not found
num_examples = dataset_config.get('data/' + dataset_name, 100)

# Print the number of examples to pass back to Bash
print(num_examples)
END
)

DATASET_NAME="data/${DATASET_NAME}"
echo "Number of examples for $DATASET_NAME: $NUM_TRAIN_EXAMPLES"

CLASSIFIER_NAME="distilbert-base-uncased"
EXP_DIR="exp_out/${DATASET_NAME}/${CLASSIFIER_NAME}"

## DISCERN PROCESS BEGINS ##

# Learn the initial classifier
CLASSIFIER_DIR="${EXP_DIR}/classifier"
mkdir -p ${CLASSIFIER_DIR}
python src/train_classifier.py -c config/classifier/clf_bert.json -k dataset="${DATASET_NAME}" \
    num_train_examples=${NUM_TRAIN_EXAMPLES} exp_dir="${CLASSIFIER_DIR}" mkdir=False

# Construct explanations for misclassified clusters
EXPLANATIONS_DIR="${EXP_DIR}/refine"
mkdir -p ${EXPLANATIONS_DIR}
python src/llm_refine.py -c config/refine/openai_gpt35_refine.json -k dataset="${DATASET_NAME}" \
    num_train_examples=${NUM_TRAIN_EXAMPLES} classifier="${CLASSIFIER_DIR}" exp_dir="${EXPLANATIONS_DIR}" mkdir=False \
    llm_weight="gpt-3.5-turbo-0125" cluster_embeddings=openai_v3

# Generate examples based on the explanations generated for the misclassified clusters
AUGMENT_DIR="${EXP_DIR}/augment"
mkdir -p ${AUGMENT_DIR}
python src/augment.py -c config/augment/gpt35.json -k dataset="${DATASET_NAME}" classifier="${CLASSIFIER_DIR}" \
     explanations_dir="${EXPLANATIONS_DIR}" exp_dir="${AUGMENT_DIR}" mkdir=False

for seed in {0..4}; do
    RETRAIN_DIR="${EXP_DIR}/retrain/${seed}"
    mkdir -p ${RETRAIN_DIR}
    python src/retrain_classifier.py -c config/retrain/clf_bert.json -k dataset="${DATASET_NAME}" \
        num_train_examples=${NUM_TRAIN_EXAMPLES} seed=${seed} explanations_dir="${EXPLANATIONS_DIR}" \
        augment_dir="${AUGMENT_DIR}" exp_dir="${RETRAIN_DIR}" mkdir=False new_train_examples=1000

    ACTIVE_DIR="${EXP_DIR}/active_learn/${seed}"
    mkdir -p ${ACTIVE_DIR}
    python src/active_learn.py -c config/active_learn/gpt35.json -k dataset="${DATASET_NAME}" \
        num_train_examples=${NUM_TRAIN_EXAMPLES} seed=${seed} \
        explanations_dir="${EXPLANATIONS_DIR}" exp_dir="${ACTIVE_DIR}" mkdir=False
done