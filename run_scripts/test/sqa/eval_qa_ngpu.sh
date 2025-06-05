node=0
ngpu=8
st=$(($node*$ngpu))
end=$(((($node+1)*$ngpu)-1))
count=0
#PREFIX=triva_qa/triva_qa.
#PREFIX=web_questions/split${ngpu}/web_questions.
#PREFIX=LLAMA1-test-set/split${ngpu}/LLAMA1-test-set.
PREFIX=$1/split${ngpu}/eval.jsonl. #triva_qa/split${ngpu}/eval.jsonl.
#PREFIX=LLAMA1-Test-Set/split${ngpu}/llama_questions_300.jsonl.
#PREFIX=web_questions/split${ngpu}/web_questions.jsonl.
use_audio_input=$2
pids=()
for i in $(seq -f "%02g" $st $end); do
	echo rank $PREFIX$i
	# echo CUDA_VISIBLE_DEVICES=$count run_scripts/eval_qa.sh $PREFIX$i
	
	
	(
	set -x;
	CUDA_VISIBLE_DEVICES=$count run_scripts/test/sqa/eval_qa.sh $model_path $PREFIX$i $use_audio_input
	) &
	pids+=($!) # store background pids
	
	count=$(($count+1))
done
i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
[ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
