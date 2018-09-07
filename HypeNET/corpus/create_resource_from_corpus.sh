#!/usr/bin/env bash

wiki_dump_file=$1
directory=$2
resource_prefix=corpus

frequent_path_count=5

start=$3
end=$4
splitted=$(($end+$start+1))

if [ ! -d $2 ]; then
   echo "Create directory $2..."
   mkdir -p $2;
fi

# Parse wikipedia. Splitting to 20 files and running in parallel.
echo 'Splitting wikipedia corpus...'
split -d -nl/$splitted $wiki_dump_file $directory/$wiki_dump_file"_";

echo 'Parsing wikipedia...'
for x in $(eval echo {$start..$end})
do
( python parse_wikipedia.py $directory/$wiki_dump_file"_"$x $directory/$wiki_dump_file"_"$x"_parsed" ) &
done
wait
echo "Wiki parsed."

echo "Concat parsed files..."
triplet_file="wiki_parsed"
cat $directory/$wiki_dump_file"_"*"_parsed" > $directory/$triplet_file
echo "Files concated to $triplet_file."

echo "Create the frequent paths file (take paths that occurred approximately at least 5 times)."
# To take paths that occurred with at least 5 different pairs,"
# replace with the commented lines - consumes much more memory).
# sort -u $triplet_file | cut -f3 -d$'\t' > paths;
# awk -F$'\t' '{a[$1]++; if (a[$1] == 5) print $1}' paths > frequent_paths;
# rm paths;
echo "Create paths for parsed files."
#for x in $(eval echo {$start..$end})
#do
#( awk -v FS='\t' '{i[$3]++} END{for(x in i){print x, i[x]}}' "$directory/$(echo $wiki_dump_file)_"$x"_parsed" > $directory/paths"_"$x ) &
#( awk -v OFS='\t' '{i[$3]++} END{for(x in i){print x, i[x]}}' "$directory/$(echo $wiki_dump_file)_"$x"_parsed" > $directory/paths"_"$x ) &
#done
#wait
python count_paths.py $directory/$triplet_file $directory/paths $directory/frequent_paths $frequent_path_count
echo "Paths for parsed files created."

#echo "Concat paths..."
#cat $directory/paths_* > $directory/paths_temp;
#cat $directory/paths_temp | grep -v "$(printf '\t1$')" > $directory/frequent_paths_temp;
#awk -F$'\t' '{i[$1]+=$2} END{for(x in i){print x"\t"i[x]}}' $directory/frequent_paths_temp > $directory/paths;
#awk -F$'\t' '$2 >= 5 {print $1}' $directory/paths > $directory/frequent_paths;
#rm $directory/paths_temp
#rm $directory/frequent_paths_temp
#rm $directory/paths_*; # You can remove paths to save space, or keep it to change the threshold for frequent paths
#echo "Paths concated."

echo "Create the terms file."
awk -F$'\t' '{a[$1]++; if (a[$1] == 1) print $1}' $directory/$triplet_file > $directory/left & PIDLEFT=$!
awk -F$'\t' '{a[$2]++; if (a[$2] == 1) print $2}' $directory/$triplet_file > $directory/right & PIDRIGHT=$!
wait $PIDLEFT
wait $PIDRIGHT
cat $directory/left $directory/right | sort -u > $directory/terms;
#rm $directory/left $directory/right &

echo 'Creating the resource from the triplets file...'

echo "First step - create the term and path to ID dictionaries..."
python create_resource_from_corpus_1.py $directory/frequent_paths $directory/terms $directory/$resource_prefix

echo "Second step - convert the textual triplets to triplets of IDs..."
for x in $(eval echo {$start..$end})
do
( python create_resource_from_corpus_2.py "$directory/$(echo $wiki_dump_file)_"$x"_parsed" $directory/$resource_prefix ) &
done
wait

echo "Third step - use the ID-based triplet file and converts it to the '_l2r.db' file..."
for x in $(eval echo {$start..$end})
do
( awk -v OFS='\t' '{i[$0]++} END{for(x in i){print x, i[x]}}' "$directory/$(echo $wiki_dump_file)_"$x"_parsed_id" > $directory/id_triplet_file"_"$x ) &
done
wait

echo "Concat temporary id-triplet files..."
cat $directory/id_triplet_file_* > $directory/id_triplet_file_temp;

echo "Compute id-triplet files..."
for x in {0..4}
do
( gawk -F $'\t' '{ if($1%5==$x) {a[$1][$2][$3]+=$4; } } END {for (i in a) for (j in a[i]) for (k in a[i][j]) print i, j, k, a[i][j][k]}' $directory/id_triplet_file_temp > $directory/id_triplet_file_$x ) &
done
wait

echo "Concat id-triplet files..."
cat $directory/id_triplet_file_* > $directory/id_triplet_file;

echo "Remove temporary id-triplet-files..."
#rm $directory/id_triplet_file_temp
#rm $directory/id_triplet_file_*

echo "Convert to _l2r.db-file."
python create_resource_from_corpus_3.py $directory/id_triplet_file $directory/$resource_prefix

# You can delete triplet_file now and keep only id_triplet_file which is more efficient, or delete both.
echo "FINISHED"

