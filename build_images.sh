

mkdir ../prod
mkdir ../develop
git checkout main
git pull
rsync -vu * ../prod/
mkdir ../prod/additional
rsync -vu ../additional/* ../prod/additional/
git checkout develop
git pull
rsync -vu * ../develop
mkdir ../develop/additional
rsync -vu ../additional/* ../develop/additional/
git checkout main

cd ../prod
sudo docker build -t pgtesting:prod -f dockerfile . 
cd ../develop
mv dockerfile_testing dockerfile
sudo docker build -t pgtesting:testing -f dockerfile .