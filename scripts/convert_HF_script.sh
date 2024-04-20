#!/usr/bin/bash

# example: 
# bash scripts/convert_HF_script.sh  default mert config_mert_base /absolute/path/to/a/fairseq/checkpoint.pt
{
    WORKING_DIR=$HOME
    MAP_PROJ_DIR=$HOME/MERT

    CONVERT_SETTING=${1:-'default'}
    MODEL_TYPE=${2:-'mert'} # hubert mert
    config_name=${3:-'mert_base'} # option: hubert, mert_base, mert_large
    checkopint_file_path=${4:-'/absolute/path/to/a/fairseq/checkpoint.pt'}

    # if not specify, then use original model name
    # s=$(basename "${CKPT_PATH}")
    model_name=$(basename $(dirname "${checkopint_file_path}"))

    ckpt_basename=$(basename "${checkopint_file_path}")
    x="$(cut -d'_' -f2 <<<"${ckpt_basename}")"_"$(cut -d'_' -f3 <<<"${ckpt_basename}")" # xxxxxxxx.pt
    ckpt_step="$(cut -d'.' -f1 <<<"$x")"

    echo "processing ${model_name} for ${ckpt_step}"

    
    case $CONVERT_SETTING in
        default)
            FAIRSEQ_PATH=${MAP_PROJ_DIR}/src/fairseq
            project_folder=${MAP_PROJ_DIR}

            output_parent_folder=${MAP_PROJ_DIR}/data/huggingface_checkpoints_MERT_exp
            ;;
        *)
            echo "Unknown setting: $CONVERT_SETTING"
            exit 1
            ;;
    esac

    case $MODEL_TYPE in
        hubert)
            # original HuBERT
            config_path=${project_folder}/scripts/hf_config/hubert.json
            ;;
        mert)
            config_path=${project_folder}/scripts/hf_config/${config_name}.json
            custom_model_source_code=${MAP_PROJ_DIR}/mert_fairseq/models/mert
            custom_model_soft_link=${FAIRSEQ_PATH}/fairseq/models/mert
            custom_dataset_source_code=${MAP_PROJ_DIR}/mert_fairseq/data/mert_dataset.py
            custom_dataset_soft_link=${FAIRSEQ_PATH}/fairseq/data/mert_dataset.py
            custom_task_source_code=${MAP_PROJ_DIR}/mert_fairseq/tasks/mert_pretraining.py
            custom_task_soft_link=${FAIRSEQ_PATH}/fairseq/tasks/mert_pretraining.py

            echo "remove temporary file ${custom_model_soft_link}"
            unlink ${custom_model_soft_link}
            unlink ${custom_dataset_soft_link}
            unlink ${custom_task_soft_link}

            echo "temporary link to user defined models ${custom_model_soft_link}"
            ln -s ${custom_model_source_code} ${custom_model_soft_link}
            ln -s ${custom_dataset_source_code} ${custom_dataset_soft_link}
            ln -s ${custom_task_source_code} ${custom_task_soft_link}
            ;;
        *)
            echo "Unknown model type: $MODEL_TYPE"
            exit 1
            ;;
    esac
    

    long_name=HF_${model_name}_ckpt_${ckpt_step}
    output_folder=${output_parent_folder}/${long_name}
    
    echo loading from:
    echo ${checkopint_file_path}
    echo output to:
    echo ${output_folder}

    cd ${project_folder}
    {
        python -m scripts.convert_MERT_HF \
            --pytorch_dump_folder ${output_folder} \
            --checkpoint_path ${checkopint_file_path} \
            --config_path ${config_path} --not_finetuned

        echo "remove temporary file ${custom_model_soft_link}"
        unlink ${custom_model_soft_link}
        unlink  ${custom_dataset_soft_link}
        unlink ${custom_task_soft_link}
    }
    exit
}