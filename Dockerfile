FROM gkousiou/laboctave

#add your own minio details here
RUN ./mc alias set minio http://10.100.59.182:9000 newacc minio@hua2021
