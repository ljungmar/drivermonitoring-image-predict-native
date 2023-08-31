import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Injectable({
  providedIn: 'root'
})
export class ApiService {

  private ipv4 = ''; // enter your ipv4 inside the quotes
  private baseUrl = 'http://' + this.ipv4 + ':8080';

  constructor(private http: HttpClient) { }

  uploadFile(formData: FormData) {
    return this.http.post<any>(`${this.baseUrl}/upload-file`, formData);
  }
}
