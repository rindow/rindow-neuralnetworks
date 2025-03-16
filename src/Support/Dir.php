<?php
namespace Rindow\NeuralNetworks\Support;

use RecursiveIteratorIterator;
use RecursiveDirectoryIterator;
use LogicException;

class Dir
{
    public static function factory() : self
    {
        return new self();
    }

    /**
     * @return array<mixed>
     */
    public function clawl(string $path,?callable $callback=null) : array
    {
        return $this->glob($path,null,$callback);
    }

    /**
     * @return array<mixed>
     */
    public function glob(string $path, ?string $pattern,?callable $callback=null) : array
    {
        if(!file_exists($path)) {
            throw new LogicException('directory not found: '.$path);
        }
        $fileSPLObjects = new RecursiveIteratorIterator(
            new RecursiveDirectoryIterator($path)
        );
        $filenames = [];
        foreach($fileSPLObjects as $fullFileName => $fileSPLObject) {
            $filename = $fileSPLObject->getFilename();
            if (!is_dir($fullFileName)) {
                if($filename!='.' && $filename!='..') {
                    if($pattern==null || preg_match($pattern, $fullFileName)) {
                        if($callback) {
                            $f = call_user_func($callback,$fullFileName);
                            if($f!==null)
                                $filenames[] = $f;
                        }
                        else {
                            $filenames[] = $fullFileName;
                        }
                    }
                }
            }
        }
        return $filenames;
    }

    public function clear(string $path) : void
    {
        $fileSPLObjects = new RecursiveIteratorIterator(
                new RecursiveDirectoryIterator($path),
                RecursiveIteratorIterator::CHILD_FIRST
        );
        foreach($fileSPLObjects as $fullFileName => $fileSPLObject) {
            $filename = $fileSPLObject->getFilename();
            if (is_dir($fullFileName)) {
                if($filename!='.' && $filename!='..') {
                    @rmdir($fullFileName);
                }
            } else {
                @unlink($fullFileName);
            }
        }
    }
}
