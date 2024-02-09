import { Component, OnInit, ElementRef, ViewChild } from '@angular/core';
import { ApiService } from '../api-calls/api-service.component';

@Component({
  selector: 'app-file-cam-upload',
  templateUrl: './file-cam-upload.component.html',
  styleUrls: ['./file-cam-upload.component.scss'],
})

export class FileCamUploadComponent implements OnInit {
  @ViewChild('videoPlayer')
  videoPlayer!: ElementRef;
  @ViewChild('capturedImage')
  capturedImage!: ElementRef;

  selectedFile: Blob | null = null;
  captureData: string | ArrayBuffer | null = null;

  file: File | null = null;
  loading = false;
  uploadStatus: string | null = null;
  accuracy: number | null = null;
  loss: number | null = null;
  prediction: string | null = null;
  
  receivedFile: string | null = null;

  constructor(private apiService: ApiService) {}
  
  ngOnInit(): void {  }

  ngAfterViewInit(): void {
    this.initializeCamera();
  }

  initializeCamera(): void {
    navigator.mediaDevices.getUserMedia({ video: true })
      .then((stream) => {
        this.videoPlayer.nativeElement.srcObject = stream;
      })
      .catch((error) => {
        console.error('Error accessing webcam:', error);
      });
  }

  captureImage() {
    const video = this.videoPlayer.nativeElement;
    console.log(video);
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    if (video.readyState === video.HAVE_ENOUGH_DATA) {

      if (context) {
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        this.captureData = canvas.toDataURL('image/jpeg');

        canvas.toBlob((blob) => {
          this.selectedFile = blob;
        }, 'image/jpeg');
      }
    }
  }

  uploadImage(): void {
    if (this.selectedFile) {
      this.prediction = "Awaiting prediction..."
      const formData: FormData = new FormData();
      formData.append('file', this.selectedFile, 'webcam_snapshot.jpg');

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

}
