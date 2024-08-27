## Dataset 

**PCN Dataset**: You can download the processed PCN dataset from the PoinTr repository. The directory structure should be

```
│PCN/
├──train/
│  ├── complete
│  │   ├── 02691156
│  │   │   ├── 1a04e3eab45ca15dd86060f189eb133.pcd
│  │   │   ├── .......
│  │   ├── .......
│  ├── partial
│  │   ├── 02691156
│  │   │   ├── 1a04e3eab45ca15dd86060f189eb133
│  │   │   │   ├── 00.pcd
│  │   │   │   ├── 01.pcd
│  │   │   │   ├── .......
│  │   │   │   └── 07.pcd
│  │   │   ├── .......
│  │   ├── .......
├──test/
│  ├── complete
│  │   ├── 02691156
│  │   │   ├── 1d63eb2b1f78aa88acf77e718d93f3e1.pcd
│  │   │   ├── .......
│  │   ├── .......
│  ├── partial
│  │   ├── 02691156
│  │   │   ├── 1d63eb2b1f78aa88acf77e718d93f3e1
│  │   │   │   └── 00.pcd
│  │   │   ├── .......
│  │   ├── .......
├──val/
│  ├── complete
│  │   ├── 02691156
│  │   │   ├── 4bae467a3dad502b90b1d6deb98feec6.pcd
│  │   │   ├── .......
│  │   ├── .......
│  ├── partial
│  │   ├── 02691156
│  │   │   ├── 4bae467a3dad502b90b1d6deb98feec6
│  │   │   │   └── 00.pcd
│  │   │   ├── .......
│  │   ├── .......
├──PCN.json
└──category.txt
```