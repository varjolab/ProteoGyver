

mkdir ../prod
mkdir ../develop
git checkout main
git pull
cp -r * ../prod/
git checkout develop
git pull
cp -r * ../develop
git checkout main

cd ../prod
sudo docker build -t pgtesting:prod -f dockerfile . 
cd ../develop
mv dockerfile_testing dockerfile
sudo docker build -t pgtesting:testing -f dockerfile .