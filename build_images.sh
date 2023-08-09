

mkdir ../prod
mkdir ../develop
git checkout main
git pull
mv * ../prod/
git checkout develop
git pull
mv * ../develop

cd ../prod
sudo docker build -t pgtesting:prod -f dockerfile . 
cd ../develop
mv dockerfile_testing dockerfile
sudo docker build -t pgtesting:testing -f dockerfile_testing .