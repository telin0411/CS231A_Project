echo "id:"
echo $1
th eval.lua -model /mnt0/data/img-cap/models/model_id$1.t7 -image_folder /mnt0/emma/image_cap/CS231A_Project/neuraltalk2-master/testimages -num_images -1 -ranker 1
