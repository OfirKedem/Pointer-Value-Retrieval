for complexity in 0 1 2 3 4 5
do
  for size in 64 128 1024 1e4 5e4 1e5 2e5 3e5 4e5 5e5 1e6 1e7 5e7
  do
    python massive_datasets_fig12.py -s $size -m $complexity
    sleep 1
  done
done
