import { Component, Input } from '@angular/core';
import { FileItemComponent } from '../file-item/file-item.component';

@Component({
  selector: 'app-home',
  templateUrl: 'home.page.html',
  styleUrls: ['home.page.scss'],
})
export class HomePage {
  @Input() files: FileItemComponent[] = [];

  onSetFiles(files: FileItemComponent[]){
    this.files = files;
  }
  constructor() {}

}
