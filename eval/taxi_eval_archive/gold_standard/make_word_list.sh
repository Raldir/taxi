
cat $1 | awk '{for(i=1;i<=NF;i++) {print $i}}' | grep -E "[^0-9]+" | sort | uniq -u > $2
