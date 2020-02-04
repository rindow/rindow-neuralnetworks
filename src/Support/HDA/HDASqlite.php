<?php
namespace Rindow\NeuralNetworks\Support\HDA;

use InvalidArgumentException;
use OutOfBoundsException;
use RuntimeException;
use PDO;
use PDOException;
use IteratorAggregate;

/*
    $hdf = new HDASqlite('filename');
    $hdf['key'] = 'data';
    $hdf['key'] = [ 'key'=>'data','key'=>'data' ];
    $hdf['key'] = [ 'data','data' ];

*/

class HDASqlite implements HDA
{
    protected $ancestor = '';
    protected $pdo;
    protected $table = 'hda';

    public function __construct($filename=null, $mode=null)
    {
        if($filename===null)
            return;
        elseif(is_string($filename))
            $this->open($filename,$mode);
        elseif($filename instanceof PDO)
            $this->node($filename,$mode);
        else
            throw new InvalidArgumentException('Invalid parent type');
    }

    public function open(string $filename, string $mode=null)
    {
        $options = [];
        if ($mode=='r' && version_compare(PHP_VERSION,'7.3')>=0) {
            $options[PDO::SQLITE_ATTR_OPEN_FLAGS] = PDO::SQLITE_OPEN_READONLY;
        }
        $this->pdo = new PDO('sqlite:'.$filename,null,null,$options);
        $this->pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
        if($mode!='r') {
            $this->createTable();
        }
    }

    public function node(PDO $pdo, string $ancestor)
    {
        $this->ancestor = $ancestor;
        $this->pdo = $pdo;
    }

    public function close()
    {
        if($this->ancestor!=='')
            throw new RuntimeException('This node is not root');
        $this->pdo = null;
    }

    public function offsetExists($offset)
    {
        $d = $this->querySingle( $this->ancestor, $offset );
        if(!$d)
            return false;
        else
            return true;
    }

    public function offsetGet( $offset )
    {
        $d = $this->querySingle( $this->ancestor, $offset );
        if(!$d)
            throw new OutOfBoundsException('Undefined index:'.$offset);
        if($d['type']=='array') {
            return new self($this->pdo, $this->ancestor.'/'.$offset);
        } elseif($d['type']=='scalar') {
            return $d['value'];
        } else {
            throw new RuntimeException('unsupported data type');
        }
    }
    public function offsetSet( $offset , $value )
    {
        $this->deleteDir($this->ancestor, $offset);
        if(is_scalar($value)) {
            $this->upsert( $this->ancestor, $offset, 'scalar', $value );
            return;
        } elseif(is_array($value)) {
            $this->upsert( $this->ancestor, $offset, 'array', '@' );
            foreach ($value as $k => $v) {
                $hda = new self($this->pdo, $this->ancestor.'/'.$offset);
                $hda[$k] = $v;
            }
        } else {
            throw new RuntimeException('unsupported data type');
        }
    }
    public function offsetUnset( $offset )
    {
        if(!$this->offsetExists($offset))
            return;
        $this->deleteDir($this->ancestor, $offset);
    }

    public function getIterator()
    {
        return new HDASqliteIterator($this->pdo, $this->table, $this->ancestor);
    }

    protected function createTable()
    {
        $sqls = [
            "CREATE TABLE IF NOT EXISTS ".$this->table." ( ancestor TEXT, key TEXT, type TEXT, value BLOB NOT NULL)",
            "CREATE UNIQUE INDEX IF NOT EXISTS fullpath ON ".$this->table." ( ancestor, key )",
            "CREATE INDEX IF NOT EXISTS ancestor ON ".$this->table." ( ancestor )",
        ];
        foreach ($sqls as $sql) {
            $stat = $this->pdo->prepare($sql);
            if(!$stat->execute())
                throw new RuntimeException('create table error:'.$sql);
        }
    }

    protected function querySingle(string $ancestor, string $key)
    {
        $stat = $this->pdo->prepare("SELECT * FROM ".$this->table." WHERE ancestor = :ancestor AND key = :key");
        if(!$stat) {
            throw new RuntimeException('query error in '.$ancestor.'/'.$key);
        }
        if(!$stat->execute([':ancestor'=>$ancestor,':key'=>$key]))
            return false;
        $row = $stat->fetch(PDO::FETCH_ASSOC);
        if(!$row)
            $stat->closeCursor();
        return $row;
    }

    protected function upsert(string $ancestor,string $key,string $type,$value)
    {
        $stat = $this->pdo->prepare("INSERT INTO ".$this->table." (ancestor,key,type,value) VALUES (:ancestor,:key,:type,:value)");
        if(!$stat) {
            throw new RuntimeException('upsert statement error in '.$ancestor.'/'.$key);
        }
        try {
            if($stat->execute([':ancestor'=>$ancestor,':key'=>$key,':type'=>$type,':value'=>$value]))
            return true;
        } catch(PDOException $e) {
            if(strpos($e->getMessage(),'SQLSTATE[23000]: Integrity constraint violation')!==0)
                throw $e;
        }
        $stat = $this->pdo->prepare("UPDATE ".$this->table." SET ancestor=:ancestor,key=:key,type=:type,value=:value WHERE ancestor=:ancestor AND key=:key");
        if($stat->execute([':ancestor'=>$ancestor,':key'=>$key,':type'=>$type,':value'=>$value]))
            return true;
        throw new RuntimeException('upsert error in '.$ancestor.'/'.$key);
    }

    protected function delete(string $ancestor,string $key)
    {
        $stat = $this->pdo->prepare("DELETE FROM ".$this->table." WHERE ancestor=:ancestor AND key=:key");
        if($stat->execute([':ancestor'=>$ancestor,':key'=>$key]))
            return true;
        throw new RuntimeException('delete error in '.$ancestor.'/'.$key);
    }

    protected function deleteDir(string $ancestor,string $key)
    {
        $stat = $this->pdo->prepare("DELETE FROM ".$this->table." WHERE ancestor >= :from AND ancestor <= :to");
        $base = $ancestor.'/'.$key;
        if(!$stat->execute([':from'=>$base.'/ ',':to'=>$base.'/~']))
            throw new RuntimeException('deleteDir error in '.$ancestor.'/'.$key);
        $stat = $this->pdo->prepare("DELETE FROM ".$this->table." WHERE ancestor = :base");
        if(!$stat->execute([':base'=>$base]))
            throw new RuntimeException('deleteDir error in '.$ancestor.'/'.$key);
        $stat = $this->pdo->prepare("DELETE FROM ".$this->table." WHERE ancestor = :ancestor AND key = :key");
        if(!$stat->execute([':ancestor'=>$ancestor,':key'=>$key]))
            throw new RuntimeException('deleteDir error in '.$ancestor.'/'.$key);
        return true;
    }
}
