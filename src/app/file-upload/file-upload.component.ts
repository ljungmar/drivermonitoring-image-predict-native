import { Component, OnInit } from '@angular/core';
import { ApiService } from './api-service.component';


@Component({
  selector: 'app-file-upload',
  templateUrl: './file-upload.component.html',
  styleUrls: ['./file-upload.component.scss'],
})
export class FileUploadComponent  implements OnInit {
  file: File | null = null;
  loading = false;
  uploadStatus: string | null = null;
  accuracy: number | null = null;
  loss: number | null = null;
  prediction: string | null = null;
  constructor(private apiService: ApiService) { }

  ngOnInit() {}

  uploadHandler(event: any) {
    const file = event?.target?.files[0];

    if (!file) {
      console.error("No file selected.");
      return;
    }

    console.log(file);

    const formData = new FormData();
    formData.append('file', file, file.name);


    console.log("formdata: ", formData.get('file'));

    this.loading = true;

    this.apiService.uploadFile(formData).subscribe(
      (res: any) => {
        this.file = res.file; // Check if the response field name is 'file' or adjust accordingly
        this.prediction = res.prediction;

        console.log("response from server: ", this.prediction);
        this.uploadStatus = 'success';
        this.loading = false;
      },
      (err) => {
        console.error(err);
        this.uploadStatus = 'failure';
        this.loading = false;
      }
    )
  }

}
