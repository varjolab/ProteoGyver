

mkdir ../prod
mkdir ../develop
git checkout master
git pull
mv * ../prod/
git checkout develop
git pull
mv * ../develop

cd ../prod
docker build -t pgtesting:prod -f dockerfile .
cd ../develop
docker build -t pgtesting:testing -f dockerfile_testing .