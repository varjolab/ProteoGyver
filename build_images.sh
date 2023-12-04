
mkdir ../prod
git checkout main
git pull
cp -r * ../prod/

cd ../prod
sudo docker build -t pgtesting:v1.102 -f dockerfile .
rm -rf ../prod
