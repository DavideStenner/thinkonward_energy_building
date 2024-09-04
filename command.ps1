# Set your AWS S3 details
$sourceFolder = "oedi-data-lake/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/"
# Define the different release versions you want to iterate over
$releases = @("2024/comstock_amy2018_release_1", "2022/resstock_amy2018_release_1.1")
$folderToData = "/timeseries_individual_buildings/by_state/"
$destinationFolder = "data_dump/"
$stateList = @("AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "HI", "IA", "ID", "IL", "IN", "KS", "KY", "LA", "MA", "MD", "ME", "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NE", "NH", "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VA", "VT", "WA", "WI", "WV", "WY")

# Set the number of files you want to copy
$N = Read-Host "Enter Number of Home to scrape"

#reset log
"" | Out-File -FilePath "log/dumping.txt"


foreach ($release in $releases){
	
	Write-Host "Starting $release"
	"Starting $release" | Out-File -FilePath "log/dumping.txt" -Append

	if ($release -eq "2024/comstock_amy2018_release_1"){
		$upgradeId = "upgrade=32/"
		$typeBuildFolderName = "commercial"
	}
	else {
		$upgradeId = "upgrade=10/"
		$typeBuildFolderName = "residential"
	}
	
	# List and sort the objects in the source folder
	$PathCurrentFolder = "$sourceFolder$release$folderToData$upgradeId"
	
	$Folders = aws s3 ls "s3://$PathCurrentFolder" --no-sign-request | Sort-Object

	foreach ($folder in $Folders) {
		$folderName = ($folder -split '\s+')[-1]
		$stateName = ($folderName -split '=')[-1]
		$stateName = $stateName -replace '/', ''

		#they put some wrong state as NA
		if ($stateName -notin $stateList){
			Write-Host "Skipping $stateName as not a US State"
			"Skipping $stateName as not a US State" | Out-File -FilePath "log/dumping.txt" -Append
			continue
		}

		$localFolderPath = "$destinationFolder$typeBuildFolderName/$folderName"
		
		Write-Host "Starting $folderName"
		"Starting $folderName" | Out-File -FilePath "log/dumping.txt" -Append

		If (!(test-path $localFolderPath)){
			mkdir $localFolderPath
		}

		$fileList = aws s3 ls "s3://$PathCurrentFolder$folderName" --no-sign-request | Sort-Object { Get-Random }
		
		# Select the first N objects
		$firstNFiles = $fileList | Select-Object -First $N
		
		#define starting command
		$awsCommand = "aws s3 cp `"s3://$PathCurrentFolder$folderName`" $localFolderPath --recursive --no-sign-request --exclude `"*`""
		
		#add every include file
		foreach ($file in $firstNFiles) {
			$fileName = ($file -split '\s+')[-1]
			$awsCommand += " --include `"$fileName`""
		}
		Invoke-Expression $awsCommand
		
		Write-Host "done"
		"done" | Out-File -FilePath "log/dumping.txt" -Append

		Start-Sleep -Seconds 10

	}
}
