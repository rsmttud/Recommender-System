echo docker image is loading... This may take a while
docker load --input=rs_export.tar
echo docker image loaded
docker run --name rs -p 80:80 rs