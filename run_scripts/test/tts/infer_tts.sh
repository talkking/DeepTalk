node=0
ngpu=8
st=$(($node*$ngpu))
end=$(((($node+1)*$ngpu)-1))
count=0
pids=()
for i in $(seq $st $end); do
	echo rank $i
	# echo CUDA_VISIBLE_DEVICES=$count run_scripts/eval_qa.sh $PREFIX$i
    	
    (
	   set -x; 
	   CUDA_VISIBLE_DEVICES=$i bash run_scripts/test/tts/infer_tts_ngpu.sh $i $ngpu &
	)
	pids+=($!) # store background pids
	
done
i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
[ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
