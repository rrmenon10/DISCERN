CONDA_BASE_PATH=$(conda info --base)
source "$CONDA_BASE_PATH/etc/profile.d/conda.sh"

conda activate discern
export MODELDEBUG_ROOT=`pwd`
export HF_HOME=${MODELDEBUG_ROOT}'/hf_cache'
export PYTHONPATH=$MODELDEBUG_ROOT:$PYTHONPATH
export PYTHON_EXEC=python
export TOKENIZERS_PARALLELISM="false"
export OPENAI_API_KEY=""