import { Component, OnInit, ElementRef, ViewChild, HostListener } from '@angular/core';
import { ApiService } from '../api-calls/api-service.component';

@Component({
  selector: 'app-desktop-view',
  templateUrl: './desktop-view.component.html',
  styleUrls: ['./desktop-view.component.scss'],
})
export class DesktopViewComponent  implements OnInit {
  @ViewChild('capturedImage')
  capturedImage!: ElementRef;

  selectedFile: Blob | null = null;
  receivedData: string | ArrayBuffer | null = null;

  file: File | null = null;
  loading = false;
  uploadStatus: string | null = null;
  prediction: string | null = null;

  isWidthLessThanHeight: boolean;
  
  constructor(private apiService: ApiService) {
    this.isWidthLessThanHeight = window.innerWidth < window.innerHeight;

  }

  @HostListener('window:resize', ['$event'])
  onResize() {
    this.isWidthLessThanHeight = window.innerWidth < window.innerHeight;
  }
  
  ngOnInit(): void {
    // Call the function every 20 seconds (20000 milliseconds)
    setInterval(() => {
      this.getImage();
    }, 10000);
  }

  getImage() {
    let rawData: any;
    this.apiService.getFile().subscribe(
      (req: any) => {
        rawData = req.last_value;
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        const img = new Image();
        
        img.src = "data:image/jpeg;base64," + rawData;
      
        // Draw the image onto the canvas once it's loaded
        img.onload = () => {
          if (context) {
            context.drawImage(img, 0, 0, canvas.width, canvas.height);
            this.receivedData = canvas.toDataURL('image/jpeg');
      
            canvas.toBlob((blob) => {
              this.selectedFile = blob;
            }, 'image/jpeg');
          }
        };
        this.uploadImage();
      },
      (err) => {
        console.error(err);
      }
      );
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
