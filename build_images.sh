

mkdir ../prod
mkdir ../develop

mkdir ../oldversion
git checkout main
git pull
cp -r * ../prod/
git checkout develop
git pull
cp -r * ../develop
git checkout old_version
git pull
cp -r * ../oldversion

cd ../prod
sudo docker build -t pgtesting:prod -f dockerfile .
cd ../develop
sudo docker build -t pgtesting:testing -f dockerfile .
cd ../old_version
sudo docker build -t pgtesting:oldversion -f dockerfile .
