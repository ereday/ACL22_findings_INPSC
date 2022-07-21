for COUNTRY in "finland" "germany" "hungary" "turkey" "united_kingdom"
do
	python download_csvs.py $COUNTRY
	mkdir -r data/$COUNTRY
	mv *.csv data/$COUNTRY
	python manifesto_data_reader.py $COUNTRY
done	

for PERCENTAGE in "25" "50" "100"
do
	python data_converter.py $PERCENTAGE
done 

echo "Done!"