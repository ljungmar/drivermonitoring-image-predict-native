import { Component, Input, OnInit } from "@angular/core";


interface FileItem {
  name: string;
}

@Component({
  selector: 'app-file-item',
  templateUrl: './file-item.component.html',
  styleUrls: ['./file-item.component.scss'],
})
export class FileItemComponent implements OnInit {

  @Input() file!: FileItem;
  @Input() directoryName: string | null = null;


  constructor() { }
  ngOnInit() { }
}
