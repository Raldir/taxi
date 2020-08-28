FILE_INPUT=$(basename "$1")
LOG=logs/predicted_$FILE_INPUT
OUTPUT=predictions/$FILE_INPUT"_predicted"
WRITE_PREDICTIONS=all

if [[ "$2" != "" ]]; then
	WRITE_PREDICTIONS="$2"
fi

echo "===================================================================================================="
echo "Starting path based and integrated predictions with input file: $1"
echo "Write output to: $OUTPUT "
echo "Write log to: $LOG"
echo "Write '$WRITE_PREDICTIONS' predicted lines."

echo "===================================================================================================="
echo "Start path based prediction..."
./taxonomy.py -c corpus/corpus_full/ -mp model_full_path_based --path_based true prediction \
	--write_predictions $WRITE_PREDICTIONS --csv_tuple_start_index=1 -i $1 -o $OUTPUT"_path_based.csv" > $LOG"_path_based.log"

echo "===================================================================================================="
echo "Start integrated prediction..."
./taxonomy.py -c corpus/corpus_full/ -mp model_full_integrated --path_based false prediction \
	--write_predictions $WRITE_PREDICTIONS --csv_tuple_start_index=1 -i $1 -o $OUTPUT"_integrated.csv" > $LOG"_integrated.log"

echo "===================================================================================================="
echo "FINISHED."
