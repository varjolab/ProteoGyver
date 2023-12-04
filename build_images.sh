
mkdir ../prod
git checkout main
git pull
cp -r * ../prod/

cd ../prod
sudo docker build -t pgtesting:1.101 -f dockerfile .
rm -rf ../prod
