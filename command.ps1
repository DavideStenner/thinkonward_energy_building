# Set your AWS S3 details
$sourceFolder = "oedi-data-lake/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2021/"
# Define the different release versions you want to iterate over
$releases = @("comstock_amy2018_release_1", "resstock_amy2018_release_1")
$folderToData = "/timeseries_individual_buildings/by_state/upgrade=0/"
$destinationFolder = "data_dump/"

# Set the number of files you want to copy
$N = 100  # Change this to the number of files you want to copy


foreach ($release in $releases){
	
	Write-Host "Starting $release"
	
	# List and sort the objects in the source folder
	$PathCurrentFolder = "$sourceFolder$release$folderToData"
	
	$Folders = aws s3 ls "s3://$PathCurrentFolder" --no-sign-request | Sort-Object | Select-Object -First 2

	foreach ($folder in $Folders) {
		$folderName = ($folder -split '\s+')[-1]
		$localFolderPath = "$destinationFolder$release/$folderName"
		
		Write-Host "Starting $folderName"
		
		If (!(test-path $localFolderPath)){
			md $localFolderPath
		}

		$fileList = aws s3 ls "s3://$PathCurrentFolder$folderName" --no-sign-request | Sort-Object { Get-Random }
		
		# Select the first N objects
		$firstNFiles = $fileList | Select-Object -First $N
		
		$awsCommand = "aws s3 cp `"s3://$PathCurrentFolder$folderName`" $localFolderPath --recursive --no-sign-request --exclude `"*`""
		
		foreach ($file in $firstNFiles) {
			$fileName = ($file -split '\s+')[-1]
			$awsCommand += " --include `"$fileName`""
		}
		Invoke-Expression $awsCommand
		
		Write-Host "done"
		Start-Sleep -Seconds 10

	}
}
