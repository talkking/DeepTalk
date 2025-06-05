input=generated/lucy_deepseek_adaptive_s3-checkpoint-22800-times3/en/train.100.clean/split8
lang=en

PWD=`dirname $0`

for gpu in `seq 0 7`; do
  echo rank $gpu
  CUDA_VISIBLE_DEVICES=$gpu python $PWD/run_wer_deeptalk.py $input/$gpu/hyp.jsonl $input/$gpu/wav_res $lang &
done
