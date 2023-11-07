

mkdir ../prod
mkdir ../develop
git checkout main
git pull
cp -r * ../prod/
git checkout develop
git pull
cp -r  * ../develop

cd ../prod
docker build -t pgtesting:prod -f dockerfile .
cd ../develop
docker build -t pgtesting:testing -f dockerfile_testing .