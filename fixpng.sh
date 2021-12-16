find train2/neg/ -name "*" -type f > list.txt

while read p; do
  name=$(echo "$p" | cut -f 1 -d '.')
  pngfix --suffix="fixed" "$p"
  rm "$p"
done <list.txt

find train2/neg/ -name "*.pngfixed" -type f > list.txt

while read p; do
  name=$(echo "$p" | cut -f 1 -d '.')
  mv "$p" "$name.png"
done <list.txt